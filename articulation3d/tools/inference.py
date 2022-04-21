import argparse
import json
import numpy as np
import os
import torch
import torch.nn.functional as F
from collections import defaultdict
import cv2
from tqdm import tqdm
import pickle
import imageio
import random
import math
from glob import glob

import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import pairwise_iou, pairwise_ioa
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask, ColorMode
from detectron2.config import get_cfg

import pytorch3d
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.utils import ico_sphere

import articulation3d.modeling  # noqa
from articulation3d.data import PlaneRCNNMapper
from articulation3d.data.planercnn_transforms import axis_to_angle_offset, angle_offset_to_axis
from articulation3d.visualization.unit_vector_plot import get_normal_figure
from articulation3d.evaluation import ArtiEvaluator
from articulation3d.config import get_planercnn_cfg_defaults
from articulation3d.utils.vis import get_pcd, project2D, random_colors, get_single_image_mesh_arti
from articulation3d.utils.mesh_utils import save_obj, get_camera_meshes, transform_meshes, rotate_mesh_for_webview
from articulation3d.utils.opt_utils import track_planes, optimize_planes
from articulation3d.utils.arti_vis import create_instances, PlaneRCNN_Branch, draw_pred, draw_gt, get_normal_map
from articulation3d.utils.visualizer import ArtiVisualizer


def save_obj_model(args, preds, frames, frame_id, axis_dir='l'):
    # for the most confident box of the first frame, visualize future frames
    p_instance = preds[frame_id]
    if p_instance.scores.shape[0] == 0:
        print("no prediction!")
        return

    box_id = p_instance.scores.argmax()
    vis = ArtiVisualizer(frames[frame_id])
    im = frames[frame_id]
    
    # computing the rotation axis
    pred_mask = p_instance.pred_masks[box_id]
    pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
    pred_plane[:, [1,2]] = pred_plane[:, [2, 1]]
    pred_plane[:, 1] = - pred_plane[:, 1]
    pred_box_centers = p_instance.pred_boxes.get_centers()

    pts = angle_offset_to_axis(p_instance.pred_rot_axis, pred_box_centers)
    verts = pred_mask.nonzero().flip(1)
    normal = F.normalize(pred_plane, p=2)[0]
    offset = torch.norm(pred_plane, p=2)
    verts_axis = pts[box_id].reshape(-1, 2)
    verts_axis_3d = get_pcd(verts_axis, normal, offset)
    if args.webvis:
        # 3d transformation for model-viewer
        verts_axis_3d = torch.tensor((np.array([[-1,0,0], [0,1,0], [0,0,-1]])@np.array([[-1,0,0],[0,-1,0],[0,0,1]])@verts_axis_3d.numpy().T).T)
    dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
    dir_vec = dir_vec / np.linalg.norm(dir_vec)

    # create visualization for rot axis
    axis_scale = pytorch3d.transforms.Scale(0.1).cuda()
    axis_pt1_t = pytorch3d.transforms.Transform3d().translate(verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
    axis_pt1_t = axis_pt1_t.cuda()
    axis_pt1 = ico_sphere(0).cuda()
    axis_pt1.verts_list()[0] = axis_scale.transform_points(axis_pt1.verts_list()[0])
    axis_pt1.verts_list()[0] = axis_pt1_t.transform_points(axis_pt1.verts_list()[0])
    axis_pt2_t = pytorch3d.transforms.Transform3d().translate(verts_axis_3d[1][0], verts_axis_3d[1][1], verts_axis_3d[1][2])
    axis_pt2_t = axis_pt2_t.cuda()
    axis_pt2 = ico_sphere(0).cuda()
    axis_pt2.verts_list()[0] = axis_scale.transform_points(axis_pt2.verts_list()[0])
    axis_pt2.verts_list()[0] = axis_pt2_t.transform_points(axis_pt2.verts_list()[0])
    axis_verts_rgb = torch.ones_like(verts)[None].cuda()  # (1, V, 3)
    axis_textures = Textures(verts_uvs=axis_verts_rgb, faces_uvs=axis_pt1.faces_list(), maps=torch.zeros((1,5,5,3)).cuda())
    axis_pt1.textures = axis_textures
    axis_pt2.textures = axis_textures
    
    # computing pcd
    plane_params = p_instance.pred_planes[box_id:(box_id + 1)]
    segmentations = p_instance.pred_masks[box_id:(box_id + 1)]
    reduce_size = False
    height = 480
    width = 640

    # bkgd mesh
    mesh_bkgd, uv_maps_bkgd = get_single_image_mesh_arti(plane_params, 1 - segmentations, img=im, height=height, width=width, webvis=args.webvis, reduce_size=reduce_size)
    #basename = 'bkgd'
    #save_obj(output_dir, basename+'_pred', mesh_bkgd, decimal_places=10, uv_maps=uv_maps_bkgd)

    # obj mesh
    mesh, uv_maps = get_single_image_mesh_arti(plane_params, segmentations, img=im, height=height, width=width, webvis=args.webvis, reduce_size=reduce_size)
    mesh = mesh.cuda()
    mesh_pcd = mesh.verts_list()[0].clone()
    
    # bitmask pcd
    pcd = get_pcd(verts, normal, offset)#, focal_length)
    pcd = pcd.float().cuda()

    # transforms
    t1 = pytorch3d.transforms.Transform3d().translate(verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
    t1 = t1.cuda()
    if axis_dir == 'l':
        angles = torch.FloatTensor(np.arange(-1.8, 0.1, 1.8/4)[:, np.newaxis])
    elif axis_dir == 'r':
        angles = torch.FloatTensor(np.arange(0.0, 1.8, 1.8/4)[:, np.newaxis])
    else:
        raise NotImplementedError
    axis_angles = angles * dir_vec
    rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
    t2 = pytorch3d.transforms.Rotate(rot_mats)
    t2 = t2.cuda()
    t3 = t1.inverse()
    pcd_trans = t3.transform_points(pcd)
    pcd_trans = t2.transform_points(pcd_trans)
    pcd_trans = t1.transform_points(pcd_trans)

    mesh_pcd_trans = t3.transform_points(mesh_pcd)
    mesh_pcd_trans = t2.transform_points(mesh_pcd_trans)
    mesh_pcd_trans = t1.transform_points(mesh_pcd_trans)

    meshes = [mesh]
    uv_maps_list = [uv_maps[0]]
    for i in range(mesh_pcd_trans.shape[0]):
        faces_list = mesh.faces_list()
        new_faces_list = [f.clone() for f in faces_list]
        imesh = Meshes(verts=[mesh_pcd_trans[i]], faces=new_faces_list, textures=mesh.textures.clone())
        meshes.append(imesh)
        uv_maps_list.append(uv_maps[0])

    # add rot_axis
    meshes.append(axis_pt1)
    meshes.append(axis_pt2)
    uv_maps_list.append(uv_maps[0])
    uv_maps_list.append(uv_maps[0])

    meshes = pytorch3d.structures.join_meshes_as_batch(meshes)
    meshes = meshes.cpu()
    # blend color for uv_maps
    for i in range(5):
        color = np.array([[[252/255, 116/255, 81/255]]]) * (i/10 + 1/2)
        uv_maps_list[i] = (uv_maps_list[i] / 255.0 + color) / 2
        uv_maps_list[i] = (uv_maps_list[i] * 255.0).astype(np.uint8) 

    uv_maps_list[-1] = (uv_maps_list[-1] / 255.0 + np.array([[[56/255, 207/255, 252/255]]])) / 2
    uv_maps_list[-1] = (uv_maps_list[-1] * 255.0).astype(np.uint8) 
    uv_maps_list[-2] = (uv_maps_list[-2] / 255.0 + np.array([[[56/255, 207/255, 252/255]]])) / 2
    uv_maps_list[-2] = (uv_maps_list[-2] * 255.0).astype(np.uint8) 

    meshes = pytorch3d.structures.join_meshes_as_batch([meshes, mesh_bkgd])
    uv_maps_list = uv_maps_list + uv_maps_bkgd

    output_dir = os.path.join(args.output, 'frame_{:0>4}'.format(frame_id))
    os.mkdir(output_dir)
    basename = 'arti'
    save_obj(output_dir, basename+'_pred', meshes, decimal_places=10, uv_maps=uv_maps_list)


def main():
    random.seed(2020)
    np.random.seed(2020)

    # command line arguments
    parser = argparse.ArgumentParser(
        description="A script that generates results of articulation prediction."
    )
    parser.add_argument("--config", required=True, help="config/config.yaml")
    parser.add_argument("--input", required=True, help="input video file")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument('--save-obj', action='store_true')
    parser.add_argument('--webvis', action='store_true')
    parser.add_argument("--conf-threshold", default=0.7, type=float, help="confidence threshold")
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output, exist_ok=True)

    # setup logger
    logger = setup_logger()

    # load model
    cfg = get_cfg()
    get_planercnn_cfg_defaults(cfg)
    cfg.merge_from_file(args.config)
    model = PlaneRCNN_Branch(cfg)
    shortened_class_names = {'arti_rot':'R', 'arti_tran':'T'}
    metadata = MetadataCatalog.get('arti_train')
    cls_name_map = [shortened_class_names[cls] for cls in metadata.thing_classes]
    
    # read video and run per-frame inference
    video_path = args.input
    is_video = True
    if video_path.endswith('mp4'):
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
    elif video_path.endswith('png'):  # accept images as input
        is_video = False
        reader = [imageio.imread(video_path)]
    
    frames = []
    preds = []
    org_vis_list = []
    for i, im in enumerate(tqdm(reader)):
        im = cv2.resize(im, (640, 480))
        frames.append(im)
        im = im[:, :, ::-1]
        pred = model.inference(im)
        pred_dict = model.process(pred)
        p_instance = create_instances(
            pred_dict['instances'], im.shape[:2], 
            pred_planes=pred_dict['pred_plane'].numpy(), 
            pred_rot_axis=pred_dict['pred_rot_axis'],
            pred_tran_axis=pred_dict['pred_tran_axis'],
            conf_threshold=args.conf_threshold,
        )
        preds.append(p_instance)

        # visualization without optmization
        if args.output is not None:
            vis = ArtiVisualizer(im[:, :, ::-1])
            seg_pred = draw_pred(vis, p_instance, metadata, cls_name_map, conf_threshold=args.conf_threshold)

            # surface normal
            if len(p_instance.pred_boxes) == 0:
                normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, 480, 640))
            else:
                normal_vis = get_normal_map(p_instance.pred_planes, p_instance.pred_masks.cpu())

            # combine visualization and generate output
            combined_vis = np.concatenate((seg_pred, normal_vis), axis=1)
            org_vis_list.append(combined_vis)

    if is_video:
        reader.close()

    # temporal optimization
    planes = track_planes(preds)
    opt_preds = optimize_planes(preds, planes, '3dc', frames=frames)

    # video visualization in 2D
    if is_video:
        writer = imageio.get_writer(os.path.join(args.output, '{}.mp4'.format('output')), fps=fps)
    else:
        write_path = os.path.join(args.output, '{}.png'.format('output'))
    
    for i, im in (enumerate(frames)):
        p_instance = opt_preds[i]
        org_vis = org_vis_list[i]

        vis = ArtiVisualizer(im)

        seg_pred = draw_pred(vis, p_instance, metadata, cls_name_map)

        # surface normal
        if len(p_instance.pred_boxes) == 0:
            normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, 480, 640))
        else:
            normal_vis = get_normal_map(p_instance.pred_planes, p_instance.pred_masks.cpu())

        # combine visualization and generate output
        combined_vis = np.concatenate((seg_pred, normal_vis, org_vis), axis=1)
        
        if is_video:
            writer.append_data(combined_vis)
        else:
            imageio.imwrite(write_path, combined_vis)

    if args.save_obj:
        # select frame_ids you want to visualize
        frame_ids = [0, 30, 60, 89]
        for frame_id in frame_ids:
            save_obj_model(args, opt_preds, frames, frame_id)


if __name__ == "__main__":
    main()
