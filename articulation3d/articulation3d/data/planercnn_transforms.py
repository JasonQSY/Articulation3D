# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import json
import logging
import numpy as np
import os
import torch
import pickle
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import (
    BitMasks,
    Boxes, 
    BoxMode, 
    Instances, 
    PolygonMasks,
    polygons_to_bitmask,
)
import cv2
# from pytorch3d.io import load_obj

# from meshrcnn.structures import MeshInstances, VoxelInstances
# from meshrcnn.utils import shape as shape_utils

from PIL import Image

__all__ = ["PlaneRCNNMapper"]


def axis_to_angle_offset(axis, center, mine=False):
    """
    Ax + By + C = 0
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1

    x cos + y sin = p, 
    where p is distance from origin to line;
    theta is angle between line and y-axis(line -> y; CCW).

    return [sin(theta), cos(theta), offset, valid]
    """
    
    axis_tensor, valid = [], []
    for a in axis:
        if a is None:
            axis_tensor.append([0,0,1,1])
            valid.append([0])
        else:
            axis_tensor.append(a)
            valid.append([1])
    axis_tensor = torch.FloatTensor(axis_tensor)
    valid = torch.FloatTensor(valid)
    axis_tensor = axis_tensor - torch.cat((center, center), dim=1)
    x1, y1, x2, y2 = axis_tensor[:,:1], axis_tensor[:,1:2], axis_tensor[:,2:3], axis_tensor[:,3:4]
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    lll = torch.sqrt(A*A + B*B)
    offset = torch.abs(C) / lll / 100  # 800 = diag(480, 640)
    if mine:
        cos = - A / lll #* torch.sign(C) / lll
        sin = - B / lll # * torch.sign(C) / lll
    else:
        cos = - A * torch.sign(C) / lll
        sin = - B * torch.sign(C) / lll
    return torch.cat((sin,cos,offset,valid), dim=1)


# def angle_offset_to_axis(angle_offsets, centers):
#     """
#     x cos + y sin = p, 
#     where p is distance from origin to line;
#     theta is angle between line and y-axis(line -> y; CCW).

#     return [x1,y1,x2,y2]
#     TODO: follow https://github.com/Hanqer/deep-hough-transform/blob/master/basic_ops.py#L174
#     """
#     rtn = []
#     for angle_offset, center in zip(angle_offsets, centers):
#         sin, cos, p = angle_offset
#         p = p * 100 # 800 = diag(480, 640)
#         x0, y0 = center
#         if cos != 0 and sin != 0:
#             yx0 = (p + x0*cos) / sin + y0
#             yx640 = (p + (x0-640)*cos) / sin + y0
#             rtn.append([
#                 0, yx0, 640, yx640
#             ])
#         elif cos == 0:
#             rtn.append([
#                 0, p / sin + y0, 640, p / sin + y0, 
#             ])
#         else:
#             rtn.append([
#                 p / cos + x0, 0, p / cos + x0, 480 
#             ])
#     return torch.tensor(rtn).long()

def angle_offset_to_axis(angle_offsets, centers, H=480, W=640):
    """
    x cos + y sin = p, 
    where p is distance from origin to line;
    theta is angle between line and y-axis(line -> y; CCW).
    
    return [x1,y1,x2,y2]
    """
    rtn = []
    for angle_offset, center in zip(angle_offsets, centers):
        sin, cos, p = angle_offset
        p = p * 100 # 800 = diag(480, 640)
        x0, y0 = center
        if sin == 0:
            angle = - np.pi / 2
        else:
            angle = - np.arctan(cos / sin)
        x, y = p * cos + x0, p * sin + y0
        p1, p2 = get_boundary_point(y, x, angle, H, W)
        try:
            rtn.append([
                p1[0], p1[1], p2[0], p2[1]
            ])
        except Exception as e:
            print(f"bad axis x={x}, y={y}, angle={angle}")
            rtn.append([
                0,0,1,1
            ])
    return torch.tensor(rtn).long()

def get_boundary_point(y, x, angle, H, W):
    '''
    Given point y,x with angle, return a two point in image boundary with shape [H, W]
    return point:[x, y]
    '''
    point1 = None
    point2 = None
    
    if angle == -np.pi / 2:
        point1 = (x, 0)
        point2 = (x, H-1)
    elif angle == 0.0:
        point1 = (0, y)
        point2 = (W-1, y)
    else:
        k = np.tan(angle)
        if y-k*x >=0 and y-k*x < H:  #left
            if point1 == None:
                point1 = (0, int(y-k*x))
            elif point2 == None:
                point2 = (0, int(y-k*x))
                if point2 == point1: point2 = None
        # print(point1, point2)
        if k*(W-1)+y-k*x >= 0 and k*(W-1)+y-k*x < H: #right
            if point1 == None:
                point1 = (W-1, int(k*(W-1)+y-k*x))
            elif point2 == None:
                point2 = (W-1, int(k*(W-1)+y-k*x)) 
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x-y/k >= 0 and x-y/k < W: #top
            if point1 == None:
                point1 = (int(x-y/k), 0)
            elif point2 == None:
                point2 = (int(x-y/k), 0)
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x-y/k+(H-1)/k >= 0 and x-y/k+(H-1)/k < W: #bottom
            if point1 == None:
                point1 = (int(x-y/k+(H-1)/k), H-1)
            elif point2 == None:
                point2 = (int(x-y/k+(H-1)/k), H-1)
                if point2 == point1: point2 = None
        # print(int(x-y/k+(H-1)/k), H-1)
        if point2 == None : point2 = point1
    return point1, point2
    


def annotations_to_instances(annos, image_size, mask_format="polygon", max_num_planes=20):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Args:
        annos (list[dict]): a list of annotations, one per instance.
        image_size (tuple): height, width
    Returns:
        Instances: It will contains fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
    """
    boxes = [BoxMode.convert(obj["bbox"], BoxMode(obj["bbox_mode"]), BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    try:
        boxes = target.gt_boxes = Boxes(boxes)
    except:
        import pdb;pdb.set_trace()
        pass
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks
    
    if len(annos) and "plane" in annos[0]:
        plane = [torch.tensor(obj["plane"]) for obj in annos]
        plane_idx = [torch.tensor([i]) for i in range(len(plane))]
        target.gt_planes = torch.stack(plane, dim=0)
        target.gt_plane_idx = torch.stack(plane_idx, dim=0)

    if len(annos) and "rot_axis" in annos[0]:
        box_centers = target.gt_boxes.get_centers()
        target.gt_rot_axis = axis_to_angle_offset([obj['rot_axis'] for obj in annos], box_centers)
    
    if len(annos) and "tran_axis" in annos[0]:
        box_centers = target.gt_boxes.get_centers()
        target.gt_tran_axis = axis_to_angle_offset([obj['tran_axis'] for obj in annos], box_centers)
    
    return target

class PlaneRCNNMapper:
    """
    A callable which takes a dict produced by the detection dataset, and applies transformations,
    including image resizing and flipping. The transformation parameters are parsed from cfg file
    and depending on the is_train condition.

    Note that for our existing models, mean/std normalization is done by the model instead of here.
    """

    def __init__(self, cfg, is_train=True, dataset_names=None):
        # self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        self.cfg = cfg
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        # self.mask_on        = cfg.MODEL.MASK_ON
        # self.embedding_on   = cfg.MODEL.EMBEDDING_ON
        # self.voxel_on       = cfg.MODEL.VOXEL_ON
        # self.mesh_on        = cfg.MODEL.MESH_ON
        # self.zpred_on       = cfg.MODEL.ZPRED_ON
        self.depth_on       = cfg.MODEL.DEPTH_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        if '#' in cfg.INPUT.IMG_ROOT:
            self.img_roots = cfg.INPUT.IMG_ROOT.split('#')
        else:
            self.img_roots = [cfg.INPUT.IMG_ROOT]
        self._eval_gt_box = cfg.TEST.EVAL_GT_BOX
        self._freeze = cfg.MODEL.FREEZE
        # fmt: on

        if self.load_proposals:
            raise ValueError("Loading proposals not yet supported")

        self.is_train = is_train
        self.depthShift = 1000.0

        assert dataset_names is not None

    def __call__(self, dataset_dict):
        """
        Transform the dataset_dict according to the configured transformations.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a new dict that's going to be processed by the model.
                It currently does the following:
                1. Read the image from "file_name"
                2. Transform the image and annotations
                3. Prepare the annotations to :class:`Instances`
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        
        image_pad = []
        for img_root in self.img_roots:
            # dataset_dict["file_name"] = os.path.join(img_root, *dataset_dict["file_name"].split('/')[-2:])
            try:
                if not os.path.exists(dataset_dict["file_name"]):
                    dataset_dict["file_name"] = dataset_dict["file_name"].replace('.jpg', '.png')
                    
                if not os.path.exists(dataset_dict["file_name"]):
                    dataset_dict["file_name"] = dataset_dict["file_name"].replace('frames_hq', 'frames_hq_neg')
                    
                image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
                image = cv2.resize(image, (dataset_dict['width'], dataset_dict['height']))
            
            except:
                print(dataset_dict["file_name"])
                image = np.zeros((dataset_dict['height'], dataset_dict['width'], 3))
            image_pad.append(image)
        image = np.concatenate(image_pad, axis=2)
        # utils.check_image_size(dataset_dict, image)

        # image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day
        if self.depth_on and 'depth_head' not in self._freeze:
            if 'depth_path' in dataset_dict.keys():
                # load depth map
                house, img_id = dataset_dict['image_id'].split('_',1)
                depth_path = dataset_dict['depth_path']
                depth = cv2.imread(depth_path, -1).astype(np.float32) / self.depthShift
                dataset_dict["depth"] = torch.as_tensor(depth.astype("float32"))

        if not self.is_train and not self._eval_gt_box:
            # for i in range(2):
            #     dataset_dict[str(i)].pop("annotations", None)
            return dataset_dict
        if not self._eval_gt_box:
            if "annotations" in dataset_dict:
                annos = [
                    self.transform_annotations(obj)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                # Should not be empty during training
                instances = annotations_to_instances(annos, image_shape)
                dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
        else:
            if "annotations" in dataset_dict:
                annos = [
                    self.transform_annotations(obj)
                    for obj in dataset_dict["annotations"]
                    if obj.get("iscrowd", 0) == 0
                ]
                # Should not be empty during training
                instances = annotations_to_instances(annos, image_shape)
                dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]

        return dataset_dict

    def transform_annotations(self, annotation, transforms=None, image_size=None):
        """
        Apply image transformations to the annotations.
        After this method, the box mode will be set to XYXY_ABS.
        """
        return annotation

