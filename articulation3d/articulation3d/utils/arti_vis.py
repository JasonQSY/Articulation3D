import argparse
import json
import numpy as np
import os
import torch
from collections import defaultdict
import cv2
from tqdm import tqdm
import pickle
import imageio
import pdb
import torch.nn.functional as F

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

import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask, ColorMode

from articulation3d.data.planercnn_transforms import axis_to_angle_offset, angle_offset_to_axis
from articulation3d.utils.visualizer import ArtiVisualizer


class PlaneRCNN_Branch():
    def __init__(self, cfg, cpu_device="cpu"):
        self.predictor = DefaultPredictor(cfg)
        self._cpu_device = cpu_device
        self._K_inv_dot_xy_1 = torch.FloatTensor(self.get_K_inv_dot_xy_1()).to("cuda")
        
        self._refine_on = cfg.MODEL.REFINE_ON

    def inference(self, img):
        """
        input: img path.
        """
        img = torch.as_tensor(img.transpose(2, 0, 1).astype("float32"))
        with torch.no_grad():
            pred = self.predictor.model([{"image": img}])[0]
        return pred

    def process(self, output):
        prediction = {}
        if "instances" in output:
            instances = output["instances"].to(self._cpu_device)
            prediction['instances'] = instances_to_coco_json(instances, "demo")
            if hasattr(output['instances'], 'pred_plane'):
                prediction['pred_plane'] = output['instances'].pred_plane.to(self._cpu_device)
            #if hasattr(output['instances'], 'pred_axis'):
            #    prediction['pred_axis'] = output['instances'].pred_axis.to(self._cpu_device)

            if hasattr(output['instances'], 'pred_rot_axis'):
                prediction['pred_rot_axis'] = output['instances'].pred_rot_axis.to(self._cpu_device)
            else:  # make sure it's compatible with old model
                if hasattr(output['instances'], 'pred_axis'):
                    print("[warning] loading pred_axis instead of pred_rot_axis")
                    prediction['pred_rot_axis'] = output['instances'].pred_axis.to(self._cpu_device)

            if hasattr(output['instances'], 'pred_tran_axis'):
                prediction['pred_tran_axis'] = output['instances'].pred_tran_axis.to(self._cpu_device)

        if (output['depth'] is not None) and (not self._refine_on):
            prediction['pred_depth'] = output['depth'].to(self._cpu_device)
            xyz = self.depth2XYZ(output['depth'])
            prediction = self.override_depth(xyz, prediction)
        return prediction
    

    def depth2XYZ(self, depth):
        """
        Convert depth to point clouds
        X - width ->
        Y - height down+
        Z - depth x
        # TODO: switch to scannet coord
        """
        XYZ = self._K_inv_dot_xy_1 * depth
        return XYZ

    def get_K_inv_dot_xy_1(self, h=480, w=640, focal_length=571.623718):
        # intrinsics from https://github.com/princeton-vl/DeepV2D/issues/30
        offset_x = 319.5
        offset_y = 239.5

        K = [[focal_length, 0, offset_x],
            [0, focal_length, offset_y],
            [0, 0, 1]]

        K_inv = np.linalg.inv(np.array(K))

        K_inv_dot_xy_1 = np.zeros((3, h, w))

        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640
                    
                ray = np.dot(K_inv,
                            np.array([xx, yy, 1]).reshape(3, 1))
                K_inv_dot_xy_1[:, y, x] = ray[:, 0]

        return K_inv_dot_xy_1.reshape(3, h, w)

    @staticmethod
    def override_depth(xyz, instance):
        pred_masks = [p['segmentation'] for p in instance['instances']]
        # scannet 2 suncg
        plane_params = instance['pred_plane']
        plane_params[:, [1,2]] = plane_params[:, [2, 1]]
        plane_params[:, 1] = - plane_params[:, 1]

        override_list = []
        for mask, plane in zip(pred_masks, plane_params):
            bimask = mask_util.decode(mask)
            if bimask.sum() == 0:
                override_list.append(plane)
                continue
            xyz_tmp = xyz[:, torch.BoolTensor(bimask)]
            offset = np.linalg.norm(plane)
            normal = plane / max(offset, 1e-8)
            offset_new = (normal@xyz_tmp.cpu().numpy()).mean()
            override_list.append(normal*offset_new)
        if len(override_list) > 0:
            override_list = torch.stack(override_list)
            override_list[:, [1,2]] = override_list[:, [2, 1]]
            override_list[:, 2] = -override_list[:, 2]
            instance['pred_plane'] = override_list
        return instance


def create_instances(predictions, image_size, pred_planes=None, pred_rot_axis=None, pred_tran_axis=None, conf_threshold=0.7):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    labels = np.asarray([predictions[i]["category_id"] for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    if pred_planes is not None:
        ret.pred_planes = np.asarray([pred_planes[i] for i in chosen])
        ret.pred_planes = torch.FloatTensor(ret.pred_planes)
    
    if pred_rot_axis is not None:
        pred_rot_axis = pred_rot_axis[chosen]
        ret.pred_rot_axis = pred_rot_axis

    if pred_tran_axis is not None:
        pred_tran_axis = pred_tran_axis[chosen]
        ret.pred_tran_axis = pred_tran_axis

    try:
        rle_pred_masks = [predictions[i]["segmentation"] for i in chosen]
        pred_masks = []
        for rle_pred_mask in rle_pred_masks:
            mask = mask_util.decode(rle_pred_mask)
            pred_masks.append(mask)
        #pred_masks = [predictions[i]["segmentation"] for i in chosen]
        pred_masks = np.array(pred_masks)
        ret.pred_masks = torch.FloatTensor(pred_masks)
    except Exception as e:
        #raise e
        #pdb.set_trace()
        pass

    #except KeyError:
    #    pass
    return ret


def vis_surface_normal(normal):
    normal_vis = (normal + 1.0) / 2.0 * 255.0
    normal_vis = normal_vis.astype(np.uint8)
    return normal_vis


def get_normal_map(plane, mask):
    """
    visualize normal map given plane and masks
    plane.shape: [N, 3]
    mask.shape: [N, H, W]
    return [H, W, 3]
    """
    mask = (mask > 0.5).float()
    #pdb.set_trace()
    normals = F.normalize(plane, p=2)
    normal_map = torch.matmul(mask.permute(1,2,0), normals)
    normal_vis = vis_surface_normal(normal_map.numpy())
    return normal_vis


def get_pred_labeled(predictions, score_threshold, vis, assigned_colors=None, paper_img=False, cls_name_map=None):
    boxes = predictions.pred_boxes
    scores = predictions.scores
    classes = predictions.pred_classes
    chosen = (scores > score_threshold).nonzero()[0]
    boxes = boxes[chosen]
    scores = scores[chosen]
    classes = classes[chosen]
    # labels = list(range(len(predictions)))
    labels = [f'{idx}: {score:.2f}' for idx, score in enumerate(scores)]
    try:
        masks = np.asarray(predictions.pred_masks)[chosen]
        masks = [GenericMask(x, vis.output.height, vis.output.width) for x in masks]
    except:
        masks = None
    
    alpha = 0.5

    if vis._instance_mode == ColorMode.IMAGE_BW:
        vis.output.img = vis._create_grayscale_image(
            (predictions.pred_masks.any(dim=0) > 0).numpy()
        )
        alpha = 0.3
    if paper_img:
        boxes=None
        labels=None
        
    try:
        vis.overlay_instances(
            #masks=masks,
            assigned_colors=assigned_colors,
            boxes=boxes,
            labels=labels,
            alpha=alpha,
        )
    except:
        pdb.set_trace()
        pass

    seg_pred = vis.output.get_image()
    return seg_pred


def get_gt_labeled(dic, vis, assigned_colors=None, paper_img=False, cls_name_map=None):
    """
    Draw annotations/segmentaions in Detectron2 Dataset format.

    Args:
        dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

    Returns:
        output (VisImage): image object with visualizations.
    """
    annos = dic.get("annotations", None)
    if annos:
        if "segmentation" in annos[0]:
            masks = [x["segmentation"] for x in annos]
        else:
            masks = None

        boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]
        classes = [cls_name_map[x['category_id']] for x in annos]

        labels = [f'{i}: gt' for i in classes]
        if paper_img:
            labels=None
            boxes=None
        vis.overlay_instances(
            labels=labels, boxes=boxes, masks=masks, assigned_colors=assigned_colors,
        )
    return vis


def draw_gt(vis, d, metadata, cls_name_map):
    """
    Draw visualization on ArtiVisualizer based on ground truth annotation d

    vis: ArtiVisualizer
    d: ground truth annotation dict
    metadata: detectron2 dataset metadata
    cls_name_map: 

    return the visualization image
    """
    annos = d['annotations']
    if len(annos) == 0:
        return vis.output.get_image()
    assigned_colors = []
    
    for anno in annos:
        if anno['bbox_mode'] == BoxMode.XYXY_REL.value:
            anno['bbox_mode'] = BoxMode.XYXY_ABS.value
            anno['bbox'] = [
                anno['bbox'][0] * d['width'],
                anno['bbox'][1] * d['height'],
                anno['bbox'][2] * d['width'],
                anno['bbox'][3] * d['height'],
            ]
        anno['bbox_mode'] = BoxMode(anno['bbox_mode'])

        #pt = None
        axis = None
        if metadata.thing_classes[anno['category_id']] == 'arti_rot':
            tmp_c = tuple([c/255 for c in metadata.thing_colors[anno['category_id']]])
            assigned_colors.append(tmp_c)
            axis = anno['rot_axis']                

        elif metadata.thing_classes[anno['category_id']] == 'arti_tran':
            tmp_c = tuple([c/255 for c in metadata.thing_colors[anno['category_id']]])
            assigned_colors.append(tmp_c)
            axis = anno['tran_axis']
        else:
            raise NotImplementedError

        if axis is not None:
            # convert points to angle_offset representation
            bbox = anno['bbox']
            bbox_tensor = torch.FloatTensor([bbox])
            bbox_centers = (bbox_tensor[:, [0, 1]] + bbox_tensor[:, [2, 3]]) / 2
            rot_axis = axis_to_angle_offset(
                [axis], 
                bbox_centers,
            )
            rot_axis = rot_axis[:, :3]

            # make bbox larger so that axis looks larger
            border_size = 20
            bbox_tensor[:, 0] = torch.max((bbox_tensor[:, 0] - border_size), 0)[0]
            bbox_tensor[:, 1] = torch.max((bbox_tensor[:, 1] - border_size), 0)[0]
            bbox_tensor[:, 2] = torch.min((bbox_tensor[:, 2] + border_size), torch.tensor(d['width']))[0]
            bbox_tensor[:, 3] = torch.min((bbox_tensor[:, 3] + border_size), torch.tensor(d['height']))[0]

            # convert angle_offset representation to points
            extension_rate = 1
            w_box = (bbox_tensor[0, 2] - bbox_tensor[0, 0]) * extension_rate
            h_box = (bbox_tensor[0, 3] - bbox_tensor[0, 1]) * extension_rate
            pts = angle_offset_to_axis(rot_axis, torch.FloatTensor([[w_box/2, h_box/2]]), H=h_box, W=w_box).float()
            pts[0, [0,2]] = pts[0, [0,2]] + bbox_tensor[0, 0] - w_box / extension_rate / 2 * (extension_rate - 1)
            pts[0, [1,3]] = pts[0, [1,3]] + bbox_tensor[0, 1] - h_box / extension_rate / 2 * (extension_rate - 1)
            pt = pts[0]

            # draw arrow on ArtiVisualizer
            vis.draw_arrow(x_data=[pt[0], pt[2]], y_data=[pt[1], pt[3]], color=tmp_c)

    vis = get_gt_labeled(d, vis, assigned_colors=assigned_colors, cls_name_map=cls_name_map)
    return vis.output.get_image()


def draw_pred(vis, p_instance, metadata, cls_name_map, conf_threshold=0.7):
    assigned_colors = []
    pred_box_centers = p_instance.pred_boxes.get_centers()

    for i in range(len(p_instance)):
        if metadata.thing_classes[p_instance.pred_classes[i]] == 'arti_rot':
            tmp_c = tuple([c/255 for c in metadata.thing_colors[p_instance.pred_classes[i]]])
            assigned_colors.append(tmp_c)

            w_box = p_instance.pred_boxes.tensor[i,2] - p_instance.pred_boxes.tensor[i,0]
            h_box = p_instance.pred_boxes.tensor[i,3] - p_instance.pred_boxes.tensor[i,1]
            extension_rate = 1
            w_box *= extension_rate 
            h_box *= extension_rate
            pts = angle_offset_to_axis([p_instance.pred_rot_axis[i]], torch.FloatTensor([[w_box/2, h_box/2]]), H=h_box, W=w_box).float()
            pts[0, [0,2]] = pts[0, [0,2]] + p_instance.pred_boxes.tensor[i, 0] - w_box / extension_rate / 2 * (extension_rate - 1)
            pts[0, [1,3]] = pts[0, [1,3]] + p_instance.pred_boxes.tensor[i, 1] - h_box / extension_rate / 2 * (extension_rate - 1)
            pt = pts[0]

        elif metadata.thing_classes[p_instance.pred_classes[i]] == 'arti_tran':
            tmp_c = tuple([c/255 for c in metadata.thing_colors[p_instance.pred_classes[i]]])
            assigned_colors.append(tmp_c)

            tmp = torch.zeros(len(p_instance.pred_tran_axis),1)
            tran_tmp = torch.cat((p_instance.pred_tran_axis, tmp), 1)

            w_box = p_instance.pred_boxes.tensor[i,2] - p_instance.pred_boxes.tensor[i,0]
            h_box = p_instance.pred_boxes.tensor[i,3] - p_instance.pred_boxes.tensor[i,1]
            extension_rate = 1
            w_box *= extension_rate 
            h_box *= extension_rate
            pts = angle_offset_to_axis(tran_tmp, torch.FloatTensor([[w_box/2, h_box/2]]), H=h_box, W=w_box).float()
            pts[0, [0,2]] = pts[0, [0,2]] + p_instance.pred_boxes.tensor[i, 0] - w_box / extension_rate / 2 * (extension_rate - 1)
            pts[0, [1,3]] = pts[0, [1,3]] + p_instance.pred_boxes.tensor[i, 1] - h_box / extension_rate / 2 * (extension_rate - 1)
            pt = pts[0]

        vis.draw_arrow(x_data=[pt[0], pt[2]], y_data=[pt[1], pt[3]], color=tmp_c)

    vis = get_pred_labeled(p_instance, conf_threshold, vis, assigned_colors=assigned_colors, cls_name_map=cls_name_map)
    return vis




def render_img(output_dir, meshes, uv_maps):
    device = torch.device("cuda:0")
    R, T = look_at_view_transform(2.7, 0, 0) 
    #cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    #R, T = look_at_view_transform(dist=10, elev=10, azim=-150)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # batch
    """
    batch_size = 20

    # Create a batch of meshes by repeating the cow mesh and associated textures. 
    # Meshes has a useful `extend` method which allows us do this very easily. 
    # This also extends the textures. 
    meshes = mesh.extend(batch_size)

    # Get a batch of viewing angles. 
    elev = torch.linspace(0, 180, batch_size)
    azim = torch.linspace(-180, 180, batch_size)

    # All the cameras helper methods support mixed type inputs and broadcasting. So we can 
    # view the camera from the same distance and specify dist=2.7 as a float,
    # and then specify elevation and azimuth angles for each viewpoint as tensors. 
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    """

    raster_settings = RasterizationSettings(
        image_size=(480, 640), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    #pdb.set_trace()
    meshes = meshes.to(device)
    images = renderer(meshes)
    for i in range(images.shape[0]):
        output_img = images[0, ..., :3].cpu().numpy()
        #pdb.set_trace()
        output_img = (output_img * 255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, 'render_{}.png'.format(i)), output_img)
    #pdb.set_trace()
    return images


