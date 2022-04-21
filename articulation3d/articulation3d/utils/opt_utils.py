
import cv2
import numpy as np
import imageio
import os
import torch
import torch.nn.functional as F
from glob import glob
import random
from scipy.stats import linregress, spearmanr
import pycocotools.mask as mask_util
import pdb

import pytorch3d
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures,
    TexturesUV,
    TexturesVertex
)

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures.boxes import pairwise_iou, pairwise_ioa
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask, ColorMode
from detectron2.structures import Boxes, BoxMode, Instances

from .vis import get_pcd, project2D
from articulation3d.visualization import get_gt_labeled_seg, get_labeled_seg
from articulation3d.data.planercnn_transforms import axis_to_angle_offset, angle_offset_to_axis
from articulation3d.utils.metrics import compare_axis, Line, EA_metric

num_planes = 0
arti_planes = 0


def fit_plane_from_normals(normals):
    """
    Copy from David: 
    
    Given Nx3 matrix S. Given a vector v 3x1, the dot Nx1 matrix of dot products is Sv.
    The sum of squares of the dot products is 
        ||Sv||_2^2 = (Sv)^T (Sv) = v^T S^T Sv = v^T (S^T S) v.
    The dot product is cos(theta), so you're minimizing cos(theta)^2.
    The unit-norm vector that maximizes that form is the largest eigenvector of (S^T S)
    The unit-norm vector that minimizes the form is the smallest eigenvector of (S^T S), 
    and the plane that is most perpendicular are given by the the two smallest eigenvectors
    
    We pick up the largest eigenvector, given that it is the normal of fitted plane.

    Args:
        normals: torch.FloatTensor of shape (N, 3)

    Returns:
        plane_n: normal of the fitted plane
    """
    STS = torch.transpose(normals, 0, 1) @ normals
    results = torch.svd(STS)
    plane_n = results.V[:, 2]
    return plane_n


def optimize_planes_average(preds, planes):
    for plane in planes:
        std_axes = []
        for idx in plane['ids']:
            box_id = plane['ids'][idx]

            # transform pred_axis to std_axis
            p_instance = preds[idx]
            pred_box_centers = p_instance.pred_boxes.get_centers()
            pts = angle_offset_to_axis(
                p_instance.pred_rot_axis, pred_box_centers)
            img_centers = torch.FloatTensor(np.array([[320, 240]]))
            std_axis = axis_to_angle_offset(pts.numpy().tolist(), img_centers)
            std_axis = std_axis[:, :3]
            std_axis = std_axis[box_id:(box_id+1)]
            std_axes.append(std_axis)

        std_axes = torch.cat(std_axes)
        std_axis = std_axes.mean(axis=0)
        plane['std_axis'] = std_axis

    opt_preds = []
    for idx, p_instance in enumerate(preds):
        pred_boxes = p_instance.pred_boxes
        chosen = [False for _ in range(pred_boxes.tensor.shape[0])]
        for plane in planes:
            if idx not in plane['ids']:
                continue
            box_id = plane['ids'][idx]
            chosen[box_id] = True
            p_instance.pred_rot_axis[box_id] = plane['std_axis']

        opt_preds.append(p_instance)

    return opt_preds


def optimize_planes_3d(preds, planes):
    for plane in planes:
        id_list = list(plane['ids'].keys())
        clusters = []
        for _ in range(5):
            # select a random frame
            select_idx = random.choice(id_list)
            box_id = plane['ids'][select_idx]
            p_instance = preds[select_idx]

            # fetch rotation axis and pcd
            pred_mask = p_instance.pred_masks[box_id]
            pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
            pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
            pred_plane[:, 1] = - pred_plane[:, 1]
            pred_box_centers = p_instance.pred_boxes.get_centers()
            pts = angle_offset_to_axis(
                p_instance.pred_rot_axis, pred_box_centers)
            verts = pred_mask.nonzero().flip(1)
            normal = F.normalize(pred_plane, p=2)[0]
            offset = torch.norm(pred_plane, p=2)
            verts_axis = pts[box_id].reshape(-1, 2)
            verts_axis_3d = get_pcd(verts_axis, normal, offset)
            dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            pcd = get_pcd(verts, normal, offset)  # , focal_length)
            pcd = pcd.float().cuda()

            # assign transformations
            t1 = pytorch3d.transforms.Transform3d().translate(
                verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
            t1 = t1.cuda()
            angles = torch.FloatTensor(
                np.arange(-np.pi/2, 0.1, np.pi/30)[:, np.newaxis])
            axis_angles = angles * dir_vec
            rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
            t2 = pytorch3d.transforms.Rotate(rot_mats)
            t2 = t2.cuda()
            t3 = t1.inverse()
            pcd_trans = t3.transform_points(pcd)
            pcd_trans = t2.transform_points(pcd_trans)
            pcd_trans = t1.transform_points(pcd_trans)

            # project pcd to 2d space
            proj_masks = []
            for i in range(pcd_trans.shape[0]):
                this_pcd = pcd_trans[i]
                proj_verts = project2D(this_pcd)
                proj_verts = proj_verts.long()
                proj_verts = proj_verts.flip(1)
                proj_mask = torch.zeros_like(pred_mask).cuda()
                proj_verts[:, 0][proj_verts[:, 0] >=
                                 proj_mask.shape[0]] = proj_mask.shape[0] - 1
                proj_verts[:, 0][proj_verts[:, 0] < 0] = 0
                proj_verts[:, 1][proj_verts[:, 1] >=
                                 proj_mask.shape[1]] = proj_mask.shape[1] - 1
                proj_verts[:, 1][proj_verts[:, 1] < 0] = 0
                full_mask = torch.ones_like(pred_mask).cuda()
                proj_mask[proj_verts[:, 0], proj_verts[:, 1]
                          ] = full_mask[proj_verts[:, 0], proj_verts[:, 1]]
                proj_mask = proj_mask.unsqueeze(0)
                proj_masks.append(proj_mask)

            proj_masks = torch.cat(proj_masks)

            # all ious
            #ious = []
            cluster_inliners = []
            cluster_angles = []
            cluster_ious = []
            for idx in id_list:
                box_id = plane['ids'][idx]
                p_instance = preds[idx]
                pred_mask = p_instance.pred_masks[box_id]
                pred_mask = pred_mask.unsqueeze(0)
                pred_mask = pred_mask.cuda()

                intersec = (pred_mask > 0.5) & (proj_masks > 0.5)
                intersec = intersec.sum(2).sum(1)
                un = (pred_mask > 0.5) | (proj_masks > 0.5)
                un = un.sum(2).sum(1)
                ious = intersec / un

                angle_id = ious.argmax()
                try:
                    angle = angles[angle_id][0]
                except:
                    pdb.set_trace()
                    pass

                if ious.max() > 0.5:
                    cluster_inliners.append(idx)
                    #id_list.remove(idx)
                    cluster_angles.append(angle)
                    cluster_ious.append(ious.max().cpu().item())

            cluster_angles = torch.FloatTensor(cluster_angles)
            cluster = {
                'center_id': select_idx,
                'inliners': cluster_inliners,
                'angles': cluster_angles,
                'ious': cluster_ious
            }
            # print(cluster)

            clusters.append(cluster)

        # now we have all clusters
        # determine the dominant cluster
        rsqs = []
        for cluster in clusters:
            if len(cluster['inliners']) < 5:
                rsqs.append(0.0)
                continue
            reg_results = linregress(
                range(cluster['angles'].shape[0]), cluster['angles'])

            #if reg_results.slope < 0.01:
            #    rsq = 0.0
            #else:
            rsq = reg_results.rvalue ** 2
            rsqs.append(rsq)

        #cluster_cnts = np.array([len(cluster['inliners']) for cluster in clusters])
        #cluster_id = cluster_cnts.argmax()
        #final_cluster = clusters[cluster_id]

        # from the cluster, infer the articulation model
        #reg_results = linregress(range(final_cluster['angles'].shape[0]), final_cluster['angles'])
        rsqs = np.array(rsqs)

        # pdb.set_trace()

        #if rsqs.max() < 0:  # impossible
        if rsqs.max() < 0.3:
            plane['has_rot'] = False
            continue
        else:
            plane['has_rot'] = True

        # then determine the regularized mask and rot axis
        try:
            final_cluster = clusters[rsqs.argmax()]
        except:
            pdb.set_trace()
            pass
        select_idx = final_cluster['center_id']
        box_id = plane['ids'][select_idx]
        p_instance = preds[select_idx]
        std_axis = p_instance.pred_rot_axis[box_id]

        # fetch rotation axis and pcd
        pred_mask = p_instance.pred_masks[box_id]
        pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
        pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
        pred_plane[:, 1] = - pred_plane[:, 1]
        pred_box_centers = p_instance.pred_boxes.get_centers()
        pts = angle_offset_to_axis(p_instance.pred_rot_axis, pred_box_centers)
        std_axis_pts = pts.clone()
        verts = pred_mask.nonzero().flip(1)
        normal = F.normalize(pred_plane, p=2)[0]
        offset = torch.norm(pred_plane, p=2)
        verts_axis = pts[box_id].reshape(-1, 2)
        verts_axis_3d = get_pcd(verts_axis, normal, offset)
        dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        pcd = get_pcd(verts, normal, offset)  # , focal_length)
        pcd = pcd.float().cuda()

        # assign transformations
        t1 = pytorch3d.transforms.Transform3d().translate(
            verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
        t1 = t1.cuda()
        angles = torch.FloatTensor(
            np.arange(-np.pi/2, 0.1, np.pi/30)[:, np.newaxis])
        axis_angles = angles * dir_vec
        rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
        t2 = pytorch3d.transforms.Rotate(rot_mats)
        t2 = t2.cuda()
        t3 = t1.inverse()
        pcd_trans = t3.transform_points(pcd)
        pcd_trans = t2.transform_points(pcd_trans)
        pcd_trans = t1.transform_points(pcd_trans)

        # project pcd to 2d space
        proj_masks = []
        for i in range(pcd_trans.shape[0]):
            this_pcd = pcd_trans[i]
            proj_verts = project2D(this_pcd)
            proj_verts = proj_verts.long()
            proj_verts = proj_verts.flip(1)
            proj_mask = torch.zeros_like(pred_mask).cuda()
            proj_verts[:, 0][proj_verts[:, 0] >=
                             proj_mask.shape[0]] = proj_mask.shape[0] - 1
            proj_verts[:, 0][proj_verts[:, 0] < 0] = 0
            proj_verts[:, 1][proj_verts[:, 1] >=
                             proj_mask.shape[1]] = proj_mask.shape[1] - 1
            proj_verts[:, 1][proj_verts[:, 1] < 0] = 0
            full_mask = torch.ones_like(pred_mask).cuda()
            proj_mask[proj_verts[:, 0], proj_verts[:, 1]
                      ] = full_mask[proj_verts[:, 0], proj_verts[:, 1]]
            proj_mask = proj_mask.unsqueeze(0)
            proj_masks.append(proj_mask)

        proj_masks = torch.cat(proj_masks)

        plane['reg_masks'] = {}
        for idx in plane['ids']:
            box_id = plane['ids'][idx]
            p_instance = preds[idx]
            pred_mask = p_instance.pred_masks[box_id]
            pred_mask = pred_mask.unsqueeze(0)
            pred_mask = pred_mask.cuda()

            intersec = (pred_mask > 0.5) & (proj_masks > 0.5)
            intersec = intersec.sum(2).sum(1)
            un = (pred_mask > 0.5) | (proj_masks > 0.5)
            un = un.sum(2).sum(1)
            ious = intersec / un
            angle_id = ious.argmax()

            plane['reg_masks'][idx] = proj_masks[angle_id].cpu()

        plane['std_axis'] = std_axis_pts

    opt_preds = []
    for idx, p_instance in enumerate(preds):
        pred_boxes = p_instance.pred_boxes
        chosen = [False for _ in range(pred_boxes.tensor.shape[0])]
        for plane in planes:
            if idx not in plane['ids']:
                continue
            box_id = plane['ids'][idx]
            if not plane['has_rot']:
                chosen[box_id] = False
                continue
            chosen[box_id] = True
            #p_instance.pred_rot_axis[box_id] = plane['std_axis']
            

            continue
            if plane['reg_masks'][idx] is not None:
                p_instance.pred_masks[box_id] = plane['reg_masks'][idx]
                # bbox
                #mask = GenericMask(plane['reg_masks'][idx].numpy(), 480, 640)
                #box_tensor = p_instance.pred_boxes.tensor
                #box_tensor[box_id] = torch.FloatTensor(mask.bbox())
                #p_instance.pred_boxes = Boxes(box_tensor)

        chosen = np.array(chosen, dtype=bool)
        no_chosen = np.logical_not(chosen)

        # soft filter
        new_instance = Instances(p_instance.image_size)
        scores = np.copy(p_instance.scores)
        #scores[chosen] = - (scores[chosen] - 1) ** 2 + 1
        scores[no_chosen] = scores[no_chosen] * 0.8
        new_instance.scores = scores
        new_instance.pred_boxes = p_instance.pred_boxes
        new_instance.pred_planes = p_instance.pred_planes
        new_instance.pred_rot_axis = p_instance.pred_rot_axis
        new_instance.pred_tran_axis = p_instance.pred_tran_axis
        new_instance.pred_masks = p_instance.pred_masks
        new_instance.pred_classes = p_instance.pred_classes

        opt_preds.append(new_instance)

    return opt_preds


def optimize_planes_3dc(preds, planes, frames=None):
    """
    optimization w/ 3d clustering
    """
    for plane in planes:
        #best_idx = -1
        #min_mean_loss = 100000

        id_list = list(plane['ids'].keys())
        clusters = []
        for _ in range(5):
            if len(id_list) == 0:
                break

            # select a random frame
            select_idx = random.choice(id_list)
            box_id = plane['ids'][select_idx]
            p_instance = preds[select_idx]

            # fetch rotation axis and pcd
            pred_mask = p_instance.pred_masks[box_id]
            pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
            pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
            pred_plane[:, 1] = - pred_plane[:, 1]
            pred_box_centers = p_instance.pred_boxes.get_centers()
            pts = angle_offset_to_axis(
                p_instance.pred_rot_axis, pred_box_centers)
            verts = pred_mask.nonzero().flip(1)
            normal = F.normalize(pred_plane, p=2)[0]
            offset = torch.norm(pred_plane, p=2)
            verts_axis = pts[box_id].reshape(-1, 2)
            verts_axis_3d = get_pcd(verts_axis, normal, offset)
            dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            pcd = get_pcd(verts, normal, offset)  # , focal_length)
            pcd = pcd.float().cuda()

            # assign transformations
            t1 = pytorch3d.transforms.Transform3d().translate(
                verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
            t1 = t1.cuda()
            # angles = torch.FloatTensor(
            #     np.arange(-np.pi/2, 0.1, np.pi/30)[:, np.newaxis])
            angles = torch.FloatTensor(
                np.arange(-np.pi/2, np.pi, np.pi/30)[:, np.newaxis]
            )
            axis_angles = angles * dir_vec
            rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
            t2 = pytorch3d.transforms.Rotate(rot_mats)
            t2 = t2.cuda()
            t3 = t1.inverse()
            pcd_trans = t3.transform_points(pcd)
            pcd_trans = t2.transform_points(pcd_trans)
            pcd_trans = t1.transform_points(pcd_trans)

            # project pcd to 2d space
            proj_masks = []
            for i in range(pcd_trans.shape[0]):
                this_pcd = pcd_trans[i]
                proj_verts = project2D(this_pcd)
                proj_verts = proj_verts.long()
                proj_verts = proj_verts.flip(1)
                proj_mask = torch.zeros_like(pred_mask).cuda()
                proj_verts[:, 0][proj_verts[:, 0] >=
                                 proj_mask.shape[0]] = proj_mask.shape[0] - 1
                proj_verts[:, 0][proj_verts[:, 0] < 0] = 0
                proj_verts[:, 1][proj_verts[:, 1] >=
                                 proj_mask.shape[1]] = proj_mask.shape[1] - 1
                proj_verts[:, 1][proj_verts[:, 1] < 0] = 0
                full_mask = torch.ones_like(pred_mask).cuda()
                proj_mask[proj_verts[:, 0], proj_verts[:, 1]
                          ] = full_mask[proj_verts[:, 0], proj_verts[:, 1]]
                proj_mask = proj_mask.unsqueeze(0)
                proj_masks.append(proj_mask)

            proj_masks = torch.cat(proj_masks)

            # all ious
            #ious = []
            cluster_inliners = []
            cluster_angles = []
            cluster_ious = []
            for idx in id_list:
                box_id = plane['ids'][idx]
                p_instance = preds[idx]
                pred_mask = p_instance.pred_masks[box_id]
                pred_mask = pred_mask.unsqueeze(0)
                pred_mask = pred_mask.cuda()

                intersec = (pred_mask > 0.5) & (proj_masks > 0.5)
                intersec = intersec.sum(2).sum(1)
                un = (pred_mask > 0.5) | (proj_masks > 0.5)
                un = un.sum(2).sum(1)
                ious = intersec / un

                angle_id = ious.argmax()
                try:
                    angle = angles[angle_id][0]
                except:
                    pdb.set_trace()
                    pass

                if ious.max() > 0.5:
                    cluster_inliners.append(idx)
                    id_list.remove(idx)
                    cluster_angles.append(angle)
                    cluster_ious.append(ious.max().cpu().item())

            cluster_angles = torch.FloatTensor(cluster_angles)
            cluster = {
                'center_id': select_idx,
                'inliners': cluster_inliners,
                'angles': cluster_angles,
                'ious': cluster_ious
            }
            # print(cluster)

            clusters.append(cluster)

        # now we have all clusters
        # determine the dominant cluster
        rsqs = []
        for cluster in clusters:
            if len(cluster['inliners']) < 5:
                rsqs.append(0.0)
                continue
            reg_results = linregress(
                range(cluster['angles'].shape[0]), cluster['angles'])
            
            rsq = reg_results.rvalue ** 2
            rsqs.append(rsq)

        rsqs = np.array(rsqs)

        #if rsqs.max() < 0:  # impossible
        if rsqs.max() < 0.3:
        #if rsqs.max() < 1.0:
            plane['has_rot'] = False
            continue
        else:
            plane['has_rot'] = True

        # then determine the regularized mask and rot axis
        try:
            final_cluster = clusters[rsqs.argmax()]
        except:
            pdb.set_trace()
            pass
        select_idx = final_cluster['center_id']
        box_id = plane['ids'][select_idx]
        p_instance = preds[select_idx]
        std_axis = p_instance.pred_rot_axis[box_id]

        # fetch rotation axis and pcd
        pred_mask = p_instance.pred_masks[box_id]
        pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
        pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
        pred_plane[:, 1] = - pred_plane[:, 1]
        pred_box_centers = p_instance.pred_boxes.get_centers()
        pts = angle_offset_to_axis(p_instance.pred_rot_axis, pred_box_centers)
        std_axis_pts = pts[box_id]
        verts = pred_mask.nonzero().flip(1)
        normal = F.normalize(pred_plane, p=2)[0]
        offset = torch.norm(pred_plane, p=2)
        verts_axis = pts[box_id].reshape(-1, 2)
        verts_axis_3d = get_pcd(verts_axis, normal, offset)
        dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        pcd = get_pcd(verts, normal, offset)  # , focal_length)
        pcd = pcd.float().cuda()

        # assign transformations
        t1 = pytorch3d.transforms.Transform3d().translate(
            verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
        t1 = t1.cuda()
        #angles = torch.FloatTensor(
        #    np.arange(-np.pi/2, 0.1, np.pi/30)[:, np.newaxis]
        #)
        angles = torch.FloatTensor(
            np.arange(-np.pi/2, np.pi/2, np.pi/30)[:, np.newaxis]
        )
        axis_angles = angles * dir_vec
        rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
        t2 = pytorch3d.transforms.Rotate(rot_mats)
        t2 = t2.cuda()
        t3 = t1.inverse()    
        #pcd_trans = t3.transform_points(pcd)
        #pcd_trans = t2.transform_points(pcd_trans)
        #pcd_trans = t1.transform_points(pcd_trans)
        trans = t3.compose(t2, t1)
        pcd_trans = trans.transform_points(pcd)
        #pdb.set_trace()
        normal_trans = trans.transform_normals(normal.cuda().unsqueeze(0))[:, 0]

        # project pcd to 2d space
        proj_masks = []
        for i in range(pcd_trans.shape[0]):
            this_pcd = pcd_trans[i]
            proj_verts = project2D(this_pcd)
            proj_verts = proj_verts.long()
            proj_verts = proj_verts.flip(1)
            proj_mask = torch.zeros_like(pred_mask).cuda()
            proj_verts[:, 0][proj_verts[:, 0] >=
                             proj_mask.shape[0]] = proj_mask.shape[0] - 1
            proj_verts[:, 0][proj_verts[:, 0] < 0] = 0
            proj_verts[:, 1][proj_verts[:, 1] >=
                             proj_mask.shape[1]] = proj_mask.shape[1] - 1
            proj_verts[:, 1][proj_verts[:, 1] < 0] = 0
            full_mask = torch.ones_like(pred_mask).cuda()
            proj_mask[proj_verts[:, 0], proj_verts[:, 1]
                      ] = full_mask[proj_verts[:, 0], proj_verts[:, 1]]
            proj_mask = proj_mask.unsqueeze(0)
            proj_masks.append(proj_mask)

        proj_masks = torch.cat(proj_masks)

        plane['reg_masks'] = {}
        plane['reg_normals'] = {}
        for idx in plane['ids']:
            box_id = plane['ids'][idx]
            p_instance = preds[idx]
            pred_mask = p_instance.pred_masks[box_id]
            pred_mask = pred_mask.unsqueeze(0)
            pred_mask = pred_mask.cuda()

            intersec = (pred_mask > 0.5) & (proj_masks > 0.5)
            intersec = intersec.sum(2).sum(1)
            un = (pred_mask > 0.5) | (proj_masks > 0.5)
            un = un.sum(2).sum(1)
            ious = intersec / un
            angle_id = ious.argmax()

            plane['reg_masks'][idx] = proj_masks[angle_id].cpu()
            
            # transform back normals
            reg_normal = normal_trans[angle_id].cpu()
            reg_normal[1] = - reg_normal[1]
            reg_normal[[1, 2]] = reg_normal[[2, 1]]
            plane['reg_normals'][idx] = reg_normal

        plane['std_axis'] = std_axis_pts

    opt_preds = []
    for idx, p_instance in enumerate(preds):
        pred_boxes = p_instance.pred_boxes
        chosen = [False for _ in range(pred_boxes.tensor.shape[0])]

        # do not filter out translation
        pred_classes = p_instance.pred_classes
        for i in range(pred_classes.size):
            if pred_classes[i] == 1:
                chosen[i] = True

        # assign optimized values back to p_instance
        p_instance.pred_rot_axis = p_instance.pred_rot_axis.clone()
        p_instance.pred_planes = p_instance.pred_planes.clone()
        for plane in planes:
            if idx not in plane['ids']:
                continue
            box_id = plane['ids'][idx]
            if not plane['has_rot']:
                chosen[box_id] = False
                continue
            chosen[box_id] = True
            pred_box_centers = pred_boxes.get_centers()[box_id:(box_id + 1)]
            std_axis = axis_to_angle_offset(
                plane['std_axis'].unsqueeze(0).numpy().tolist(),
                pred_box_centers
            )
            p_instance.pred_rot_axis[box_id] = std_axis[0, :3]

            #if plane['reg_normals'][idx] is not None:
            #    p_instance.pred_planes[box_id] = plane['reg_normals'][idx]
            continue
            if plane['reg_masks'][idx] is not None:
                p_instance.pred_masks[box_id] = plane['reg_masks'][idx]
                # bbox
                #mask = GenericMask(plane['reg_masks'][idx].numpy(), 480, 640)
                #box_tensor = p_instance.pred_boxes.tensor
                #box_tensor[box_id] = torch.FloatTensor(mask.bbox())
                #p_instance.pred_boxes = Boxes(box_tensor)

        chosen = np.array(chosen, dtype=bool)
        no_chosen = np.logical_not(chosen)

        # soft filter
        new_instance = Instances(p_instance.image_size)
        scores = np.copy(p_instance.scores)
        scores[no_chosen] = scores[no_chosen] * 0.6
        #scores[chosen] = - (scores[chosen] - 1) ** 2 + 1
        new_instance.scores = scores
        new_instance.pred_boxes = p_instance.pred_boxes
        new_instance.pred_planes = p_instance.pred_planes
        new_instance.pred_rot_axis = p_instance.pred_rot_axis
        new_instance.pred_tran_axis = p_instance.pred_tran_axis
        new_instance.pred_masks = p_instance.pred_masks
        new_instance.pred_classes = p_instance.pred_classes

        opt_preds.append(new_instance)

    return opt_preds


def optimize_planes_3d_trans(preds, planes, frames=None):
    """
    optimizatino w/ 3d clustering
    """
    for plane in planes:
        id_list = list(plane['ids'].keys())
        clusters = []
        for _ in range(5):
            if len(id_list) == 0:
                break

            # select a random frame
            select_idx = random.choice(id_list)
            box_id = plane['ids'][select_idx]
            p_instance = preds[select_idx]

            # fetch rotation axis and pcd
            pred_mask = p_instance.pred_masks[box_id]
            pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
            pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
            pred_plane[:, 1] = - pred_plane[:, 1]
            pred_box_centers = p_instance.pred_boxes.get_centers()

            axis_tran = p_instance.pred_tran_axis
            tmp = torch.zeros(len(axis_tran),1)
            axis_tran = torch.cat((axis_tran, tmp), 1)
            pts = angle_offset_to_axis(axis_tran, pred_box_centers)

            verts = pred_mask.nonzero().flip(1)
            normal = F.normalize(pred_plane, p=2)[0]
            offset = torch.norm(pred_plane, p=2)
            verts_axis = pts[box_id].reshape(-1, 2)
            verts_axis_3d = get_pcd(verts_axis, normal, offset)
            dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            pcd = get_pcd(verts, normal, offset)  # , focal_length)
            pcd = pcd.float().cuda()

            # assign transformations
            angles = torch.arange(-1, 1, 0.1).unsqueeze(1)
            trans_vectors = angles * dir_vec
            t1 = pytorch3d.transforms.Transform3d().translate(trans_vectors)
            t1 = t1.cuda()
            pcd_trans = t1.transform_points(pcd)

            # project pcd to 2d space
            proj_masks = []
            for i in range(pcd_trans.shape[0]):
                this_pcd = pcd_trans[i]
                proj_verts = project2D(this_pcd)
                proj_verts = proj_verts.long()
                proj_verts = proj_verts.flip(1)
                proj_mask = torch.zeros_like(pred_mask).cuda()
                proj_verts[:, 0][proj_verts[:, 0] >=
                                 proj_mask.shape[0]] = proj_mask.shape[0] - 1
                proj_verts[:, 0][proj_verts[:, 0] < 0] = 0
                proj_verts[:, 1][proj_verts[:, 1] >=
                                 proj_mask.shape[1]] = proj_mask.shape[1] - 1
                proj_verts[:, 1][proj_verts[:, 1] < 0] = 0
                full_mask = torch.ones_like(pred_mask).cuda()
                proj_mask[proj_verts[:, 0], proj_verts[:, 1]
                          ] = full_mask[proj_verts[:, 0], proj_verts[:, 1]]
                proj_mask = proj_mask.unsqueeze(0)
                proj_masks.append(proj_mask)

            proj_masks = torch.cat(proj_masks)

            # all ious
            #ious = []
            cluster_inliners = []
            cluster_angles = []
            cluster_ious = []
            for idx in id_list:
                box_id = plane['ids'][idx]
                p_instance = preds[idx]
                pred_mask = p_instance.pred_masks[box_id]
                pred_mask = pred_mask.unsqueeze(0)
                pred_mask = pred_mask.cuda()

                intersec = (pred_mask > 0.5) & (proj_masks > 0.5)
                intersec = intersec.sum(2).sum(1)
                un = (pred_mask > 0.5) | (proj_masks > 0.5)
                un = un.sum(2).sum(1)
                ious = intersec / un

                angle_id = ious.argmax()
                try:
                    angle = angles[angle_id][0]
                except:
                    pdb.set_trace()
                    pass

                if ious.max() > 0.5:
                    cluster_inliners.append(idx)
                    id_list.remove(idx)
                    cluster_angles.append(angle)
                    cluster_ious.append(ious.max().cpu().item())

            cluster_angles = torch.FloatTensor(cluster_angles)
            cluster = {
                'center_id': select_idx,
                'inliners': cluster_inliners,
                'angles': cluster_angles,
                'ious': cluster_ious
            }
            # print(cluster)

            clusters.append(cluster)

        # now we have all clusters
        # determine the dominant cluster
        rsqs = []
        for cluster in clusters:
            if len(cluster['inliners']) < 5:
                rsqs.append(0.0)
                continue
            reg_results = linregress(
                range(cluster['angles'].shape[0]), cluster['angles'])
            
            rsq = reg_results.rvalue ** 2
            #if reg_results.slope < 0.01:
            #    rsq = 0.0
            #else:
            #    rsq = reg_results.rvalue ** 2
            rsqs.append(rsq)

        #cluster_cnts = np.array([len(cluster['inliners']) for cluster in clusters])
        #cluster_id = cluster_cnts.argmax()
        #final_cluster = clusters[cluster_id]

        # from the cluster, infer the articulation model
        #reg_results = linregress(range(final_cluster['angles'].shape[0]), final_cluster['angles'])
        rsqs = np.array(rsqs)

        # pdb.set_trace()

        #if rsqs.max() < 0:  # impossible
        if rsqs.max() < 0.3:
            plane['has_rot'] = False
            continue
        else:
            plane['has_rot'] = True

        # then determine the regularized mask and rot axis
        try:
            final_cluster = clusters[rsqs.argmax()]
        except:
            pdb.set_trace()
            pass
        select_idx = final_cluster['center_id']
        box_id = plane['ids'][select_idx]
        p_instance = preds[select_idx]
        std_axis = p_instance.pred_tran_axis[box_id]

        
            
        # fetch rotation axis and pcd
        pred_mask = p_instance.pred_masks[box_id]
        pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
        pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
        pred_plane[:, 1] = - pred_plane[:, 1]
        pred_box_centers = p_instance.pred_boxes.get_centers()
        axis_tran = p_instance.pred_tran_axis
        tmp = torch.zeros(len(axis_tran),1)
        axis_tran = torch.cat((axis_tran, tmp), 1)
        pts = angle_offset_to_axis(axis_tran, pred_box_centers)

        verts = pred_mask.nonzero().flip(1)
        normal = F.normalize(pred_plane, p=2)[0]
        offset = torch.norm(pred_plane, p=2)
        verts_axis = pts[box_id].reshape(-1, 2)
        verts_axis_3d = get_pcd(verts_axis, normal, offset)
        dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        pcd = get_pcd(verts, normal, offset)  # , focal_length)
        pcd = pcd.float().cuda()

        # assign transformations
        angles = torch.arange(-1, 1, 0.1).unsqueeze(1)
        trans_vectors = angles * dir_vec
        t1 = pytorch3d.transforms.Transform3d().translate(trans_vectors)
        t1 = t1.cuda()
        pcd_trans = t1.transform_points(pcd)

        # project pcd to 2d space
        proj_masks = []
        for i in range(pcd_trans.shape[0]):
            this_pcd = pcd_trans[i]
            proj_verts = project2D(this_pcd)
            proj_verts = proj_verts.long()
            proj_verts = proj_verts.flip(1)
            proj_mask = torch.zeros_like(pred_mask).cuda()
            proj_verts[:, 0][proj_verts[:, 0] >=
                             proj_mask.shape[0]] = proj_mask.shape[0] - 1
            proj_verts[:, 0][proj_verts[:, 0] < 0] = 0
            proj_verts[:, 1][proj_verts[:, 1] >=
                             proj_mask.shape[1]] = proj_mask.shape[1] - 1
            proj_verts[:, 1][proj_verts[:, 1] < 0] = 0
            full_mask = torch.ones_like(pred_mask).cuda()
            proj_mask[proj_verts[:, 0], proj_verts[:, 1]
                      ] = full_mask[proj_verts[:, 0], proj_verts[:, 1]]
            proj_mask = proj_mask.unsqueeze(0)
            proj_masks.append(proj_mask)

        proj_masks = torch.cat(proj_masks)

        plane['reg_masks'] = {}
        for idx in plane['ids']:
            box_id = plane['ids'][idx]
            p_instance = preds[idx]
            pred_mask = p_instance.pred_masks[box_id]
            pred_mask = pred_mask.unsqueeze(0)
            pred_mask = pred_mask.cuda()

            intersec = (pred_mask > 0.5) & (proj_masks > 0.5)
            intersec = intersec.sum(2).sum(1)
            un = (pred_mask > 0.5) | (proj_masks > 0.5)
            un = un.sum(2).sum(1)
            ious = intersec / un
            angle_id = ious.argmax()

            plane['reg_masks'][idx] = proj_masks[angle_id].cpu()

        plane['std_axis'] = std_axis

    opt_preds = []
    for idx, p_instance in enumerate(preds):
        pred_boxes = p_instance.pred_boxes
        
        chosen = [False for _ in range(pred_boxes.tensor.shape[0])]

        # do not filter out rotation
        pred_classes = p_instance.pred_classes
        for i in range(pred_classes.size):
            if pred_classes[i] == 0:
                chosen[i] = True

        # assign optimized values back to p_instance
        for plane in planes:
            if idx not in plane['ids']:
                continue
            box_id = plane['ids'][idx]
            if not plane['has_rot']:
                chosen[box_id] = False
                continue
            chosen[box_id] = True
            p_instance.pred_tran_axis[box_id] = plane['std_axis']
            continue
            if plane['reg_masks'][idx] is not None:
                p_instance.pred_masks[box_id] = plane['reg_masks'][idx]
                # bbox
                #mask = GenericMask(plane['reg_masks'][idx].numpy(), 480, 640)
                #box_tensor = p_instance.pred_boxes.tensor
                #box_tensor[box_id] = torch.FloatTensor(mask.bbox())
                #p_instance.pred_boxes = Boxes(box_tensor)

        chosen = np.array(chosen, dtype=bool)
        no_chosen = np.logical_not(chosen)

        # soft filter
        new_instance = Instances(p_instance.image_size)
        scores = np.copy(p_instance.scores)
        scores[no_chosen] = scores[no_chosen] * 0.6
        #scores[chosen] = - (scores[chosen] - 1) ** 2 + 1
        new_instance.scores = scores
        new_instance.pred_boxes = p_instance.pred_boxes
        new_instance.pred_planes = p_instance.pred_planes
        new_instance.pred_rot_axis = p_instance.pred_rot_axis
        new_instance.pred_tran_axis = p_instance.pred_tran_axis
        new_instance.pred_masks = p_instance.pred_masks
        new_instance.pred_classes = p_instance.pred_classes

        opt_preds.append(new_instance)

    return opt_preds


def optimize_planes(preds, planes, method, frames=None):
    if method == 'average':
        return optimize_planes_average(preds, planes)
    elif method == '3d':
        return optimize_planes_3d(preds, planes)
    elif method == '3dc':
        #check_monotonic(preds, planes['rot'], 'debug', frames=frames)
        opt_preds = optimize_planes_3d_trans(preds, planes['trans'], frames=frames)
        opt_preds_2 = optimize_planes_3dc(opt_preds, planes['rot'], frames=frames)
        #pdb.set_trace()
        return opt_preds_2
    else:
        raise NotImplementedError


def check_axis(preds, opt_preds, planes, method, frames=None):
    scores_all = []
    opt_scores_all = []

    for plane in planes:
        id_list = list(plane['ids'].keys())
        rot_axes = []
        box_scores = []
        for select_idx in id_list:
            # select a random frame
            #select_idx = random.choice(id_list)
            box_id = plane['ids'][select_idx]
            p_instance = preds[select_idx]
            pred_mask = p_instance.pred_masks[box_id]
            pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
            pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
            pred_plane[:, 1] = - pred_plane[:, 1]
            pred_box_centers = p_instance.pred_boxes.get_centers()
            pts = angle_offset_to_axis(
                p_instance.pred_rot_axis, pred_box_centers)
            verts = pred_mask.nonzero().flip(1)
            normal = F.normalize(pred_plane, p=2)
            offset = torch.norm(pred_plane, p=2)
            score = p_instance.scores[box_id]
            box_scores.append(score)
            rot_axis = p_instance.pred_rot_axis[box_id:(box_id + 1)]
            rot_axis_pts = angle_offset_to_axis(rot_axis, pred_box_centers)
            rot_axes.append(rot_axis_pts)

        box_scores = torch.FloatTensor(box_scores)
        rot_axes = torch.cat(rot_axes, dim=0)
        
        def axis_distance(rot_axes):
            scores = []
            for i in range(rot_axes.shape[0]):
                for j in range(rot_axes.shape[0]):
                    if i == j:
                        continue

                    try:
                        p_coord = rot_axes[i]
                        line_i = Line([p_coord[1], p_coord[0], p_coord[3], p_coord[2]])
                        g_coord = rot_axes[j]
                        line_j = Line([g_coord[1], g_coord[0], g_coord[3], g_coord[2]])

                        score = EA_metric(line_i, line_j)
                        scores.append(score)
                    except:
                        scores.append(0.0)

            scores = torch.FloatTensor(scores)
            return scores

        scores = axis_distance(rot_axes)

        opt_rot_axes = []
        opt_box_scores = []
        for select_idx in id_list:
            # select a random frame
            #select_idx = random.choice(id_list)
            box_id = plane['ids'][select_idx]
            p_instance = opt_preds[select_idx]
            pred_mask = p_instance.pred_masks[box_id]
            pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
            pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
            pred_plane[:, 1] = - pred_plane[:, 1]
            pred_box_centers = p_instance.pred_boxes.get_centers()
            pts = angle_offset_to_axis(
                p_instance.pred_rot_axis, pred_box_centers)
            verts = pred_mask.nonzero().flip(1)
            normal = F.normalize(pred_plane, p=2)
            offset = torch.norm(pred_plane, p=2)

            score = p_instance.scores[box_id]
            opt_box_scores.append(score)

            rot_axis = p_instance.pred_rot_axis[box_id:(box_id + 1)]
            rot_axis_pts = angle_offset_to_axis(rot_axis, pred_box_centers)
            opt_rot_axes.append(rot_axis_pts)

        opt_rot_axes = torch.cat(opt_rot_axes, dim=0)
        opt_box_scores = torch.FloatTensor(opt_box_scores)
        opt_scores = axis_distance(opt_rot_axes)

        if box_scores.mean() - opt_box_scores.mean() < 0.1: 
            scores_all.extend(scores)
            opt_scores_all.extend(opt_scores)

    return scores_all, opt_scores_all


def check_monotonic(preds, opt_preds, planes, method, frames=None):
    corrs = []
    opt_corrs = []
    for plane in planes:
        id_list = list(plane['ids'].keys())
        normals = []
        for select_idx in id_list:
            # select a random frame
            #select_idx = random.choice(id_list)
            box_id = plane['ids'][select_idx]
            p_instance = preds[select_idx]
            pred_mask = p_instance.pred_masks[box_id]
            pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
            pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
            pred_plane[:, 1] = - pred_plane[:, 1]
            pred_box_centers = p_instance.pred_boxes.get_centers()
            pts = angle_offset_to_axis(
                p_instance.pred_rot_axis, pred_box_centers)
            verts = pred_mask.nonzero().flip(1)
            normal = F.normalize(pred_plane, p=2)
            offset = torch.norm(pred_plane, p=2)

            normals.append(normal)
        
        normals = torch.cat(normals, dim=0)

        opt_normals = []
        for select_idx in id_list:
            # select a random frame
            #select_idx = random.choice(id_list)
            box_id = plane['ids'][select_idx]
            p_instance = opt_preds[select_idx]
            pred_mask = p_instance.pred_masks[box_id]
            pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
            pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
            pred_plane[:, 1] = - pred_plane[:, 1]
            pred_box_centers = p_instance.pred_boxes.get_centers()
            pts = angle_offset_to_axis(
                p_instance.pred_rot_axis, pred_box_centers)
            verts = pred_mask.nonzero().flip(1)
            normal = F.normalize(pred_plane, p=2)
            offset = torch.norm(pred_plane, p=2)

            opt_normals.append(normal)
        
        opt_normals = torch.cat(opt_normals, dim=0)

        

        def recover_angles(plane_n, normals):
            # project normals to plane_normal direction
            normal_proj = torch.matmul(normals, plane_n.unsqueeze(1))

            # project normals to planes
            normal_on_planes = normals - normal_proj * plane_n
            
            # we choose the first frame as the reference frame
            reference_normal = normal_on_planes[0]

            # dot product with refernce normal will convert it to cos(theta)
            cos_theta = torch.matmul(normal_on_planes, reference_normal.unsqueeze(1))
            angles = torch.acos(cos_theta)
            return angles
    

        # plane normal, perpendicular to all lines on the plane
        plane_n = fit_plane_from_normals(normals)
        fit_scores = torch.matmul(normals, plane_n.unsqueeze(1))
        fit_scores = torch.abs(fit_scores).mean()
        #angles = recover_angles(plane_n, normals)

        opt_plane_n = fit_plane_from_normals(opt_normals)
        #opt_angles = recover_angles(opt_plane_n, opt_normals)
        opt_fit_scores = torch.matmul(opt_normals, opt_plane_n.unsqueeze(1))
        opt_fit_scores = torch.abs(opt_fit_scores).mean()
        #pdb.set_trace()

        # compute the rank correlation with timestamp
        #timestamp = torch.FloatTensor(id_list)
        #corr = spearmanr(angles[:, 0], timestamp).correlation
        #corrs.append(corr)
        corrs.append([fit_scores])
        opt_corrs.append([opt_fit_scores])

    return corrs, opt_corrs
            


def track_planes(preds):
    # tracking
    planes = {
        'rot': [],
        'trans': [],
    }

    for idx, p_instance in enumerate(preds):
        plane = {}
        pred_classes = p_instance.pred_classes
        pred_boxes = p_instance.pred_boxes
        for box_id in range(p_instance.pred_boxes.tensor.shape[0]):
            current_box = pred_boxes[box_id]

            plane_cat = 'rot'
            if pred_classes[box_id] == 1:
                plane_cat = 'trans'
                #continue

            has_overlap = False
            for plane in planes[plane_cat]:
                if idx - plane['latest_frame'] > 5:
                    continue
                plane_box = plane['bbox']
                iou = pairwise_iou(current_box, plane_box)
                if iou.item() > 0.5:
                    # record it
                    has_overlap = True
                    plane['ids'][idx] = box_id
                    plane['bbox'] = current_box
                    plane['latest_frame'] = idx
                    break

            if not has_overlap:  # create new box
                plane = {
                    'bbox': current_box,
                    'ids': {
                        idx: box_id,
                    },
                    'latest_frame': idx,
                }
                planes[plane_cat].append(plane)

    # filter short sequence
    for cat in planes:
        filter_planes = []
        for plane in planes[cat]:
            if len(plane['ids']) < 10:
                continue
            filter_planes.append(plane)
        planes[cat] = filter_planes

    return planes
