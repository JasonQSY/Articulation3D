# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import itertools
import json
import logging
import numpy as np
import pickle
import os
from collections import OrderedDict, Counter
from scipy.special import softmax
import detectron2.utils.comm as comm
import pycocotools.mask as mask_util
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou, PolygonMasks
from detectron2.utils.logger import setup_logger, create_small_table
from pycocotools.coco import COCO
from fvcore.common.file_io import PathManager, file_lock
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from .detectron2coco import convert_to_coco_dict
import articulation3d.utils.VOCap as VOCap
from articulation3d.utils.metrics import compare_planes

logger = logging.getLogger(__name__)
if not logger.isEnabledFor(logging.INFO):
    setup_logger(name=__name__)


class ScannetEvaluator(COCOEvaluator):
    """
    Evaluate object proposal, instance detection, segmentation and affinity
    outputs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self.cfg = cfg
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._device = cfg.MODEL.DEVICE
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        self._coco_api = COCO(self._to_coco(self._metadata.json_file))
        self._do_evaluation = "annotations" in self._coco_api.dataset
        self._kpt_oks_sigmas = None

        self._filter_iou = 0.7
        self._filter_score = 0.7

        self._visualize = cfg.TEST.SAVE_VIS
        self._K_inv_dot_xy_1 = torch.FloatTensor(self.get_K_inv_dot_xy_1()).to(self._device)
        self._eval_gt_box = cfg.TEST.EVAL_GT_BOX

        self._refine_on = cfg.MODEL.REFINE_ON

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

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        return tasks

    def _to_coco(self, d2json):
        assert self._output_dir
        save_json = os.path.join(self._output_dir, 'scannet_coco_'+d2json.replace('/', '_'))
        PathManager.mkdirs(os.path.dirname(save_json))
        with file_lock(save_json):
            if PathManager.exists(save_json):
                logger.warning(
                    f"Using previously cached COCO format annotations at '{save_json}'. "
                    "You need to clear the cache file if your dataset has been modified."
                )
            else:
                logger.info(f"Converting annotations of dataset '{d2json}' to COCO format ...)")
                with PathManager.open(d2json, 'r') as f:
                    d2_data = json.load(f)
                coco_data = convert_to_coco_dict(d2_data['data'], self._metadata)
                with PathManager.open(save_json, 'w') as f:
                    json.dump(coco_data, f)
        return save_json

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

    def override_depth(self, xyz, instance):
        pred_masks = [p['segmentation'] for p in instance['instances']]
        # scannet 2 suncg
        plane_params = instance['pred_plane']
        plane_params[:, [1,2]] = plane_params[:, [2, 1]]
        plane_params[:, 1] = -plane_params[:, 1]

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

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {}

            prediction['image_id'] = input['image_id']
            prediction['file_name'] = input['file_name']
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction['instances'] = instances_to_coco_json(instances, input["image_id"])
                if hasattr(output['instances'], 'pred_plane'):
                    prediction['pred_plane'] = output['instances'].pred_plane.to(self._cpu_device)
            if (output['depth'] is not None) and (not self._refine_on):
                prediction['pred_depth'] = output['depth'].to(self._cpu_device)
                xyz = self.depth2XYZ(output['depth'])
                prediction = self.override_depth(xyz, prediction)
                depth_rst = get_depth_err(output['depth'], input['depth'].to(self._device))
                prediction['depth_l1_dist'] = depth_rst.to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            gt_corrs = self._gt_corrs

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_plane(predictions)
        if "depth_l1_dist" in predictions[0]:
            self._eval_depth(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_plane(self, predictions):
        results = evaluate_for_planes(
            predictions,
            self._coco_api,
            self._metadata,
            self._filter_iou,
            device=self._device,
        )
        self._results.update(results)

    def _eval_depth(self, predictions):
        depth_l1_dist = [p['depth_l1_dist'] for p in predictions]
        result = {f"depth_l1_dist": np.mean(depth_l1_dist)}
        logger.info("Depth metrics: \n"+create_small_table(result))
        self._results.update(result)


def l1LossMask(pred, gt, mask):
    """L1 loss with a mask"""        
    return torch.sum(torch.abs(pred - gt) * mask) / torch.clamp(mask.sum(), min=1)


def get_depth_err(pred_depth, gt_depth, device=None):
    l1dist = l1LossMask(pred_depth, gt_depth, (gt_depth > 1e-4).float())
    return l1dist


def angle_error_vec(v1, v2):
    return 2*np.arccos(np.clip(np.abs(np.sum(np.multiply(v1, v2), axis=1)), -1.0, 1.0))*180/np.pi

def evaluate_for_planes(predictions, dataset, metadata, filter_iou, iou_thresh=0.5, normal_threshold=30, offset_threshold=0.3, device=None):
    if device is None:
        device = torch.device("cpu")
    # classes
    cat_ids = sorted(dataset.getCatIds())
    reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}

    # initialize tensors to record box & mask AP, number of gt positives
    box_apscores, box_aplabels = {}, {}
    mask_apscores, mask_aplabels = {}, {}
    plane_apscores, plane_aplabels = {}, {}
    plane_offset_errs, plane_normal_errs = [], []
    npos = {}
    for cat_id in cat_ids:
        box_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        box_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        mask_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        mask_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        plane_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        plane_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        npos[cat_id] = 0.0

    # number of gt positive instances per class
    for gt_ann in dataset.dataset["annotations"]:
        gt_label = gt_ann["category_id"]
        npos[gt_label] += 1.0

    for prediction in predictions:
        original_id = prediction['image_id']
        image_width = dataset.loadImgs([original_id])[0]["width"]
        image_height = dataset.loadImgs([original_id])[0]["height"]
        if "instances" not in prediction:
            continue

        num_img_preds = len(prediction["instances"])
        if num_img_preds == 0:
            continue

        # predictions
        scores, boxes, labels, masks_rles = [], [], [], []
        for ins in prediction['instances']:
            scores.append(ins['score'])
            boxes.append(ins['bbox'])
            labels.append(ins['category_id'])
            masks_rles.append(ins['segmentation'])
        boxes = np.array(boxes)  # xywh from coco
        boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        boxes = Boxes(torch.tensor(np.array(boxes))).to(device)
        planes = prediction['pred_plane']

        # ground truth
        # anotations corresponding to original_id (aka coco image_id)
        gt_ann_ids = dataset.getAnnIds(imgIds=[original_id])
        gt_anns = dataset.loadAnns(gt_ann_ids)
        # get original ground truth mask, box, label & mesh
        gt_boxes, gt_labels, gt_mask_rles, gt_planes = [], [], [], []
        for ann in gt_anns:
            gt_boxes.append(ann['bbox'])
            gt_labels.append(ann['category_id'])
            if isinstance(ann['segmentation'], list):
                polygons = [np.array(p, dtype=np.float64) for p in ann['segmentation']]
                rles = mask_util.frPyObjects(polygons, image_height, image_width)
                rle = mask_util.merge(rles)
            elif isinstance(ann['segmentation'], dict):  # RLE
                rle = ann['segmentation']
            else:
                raise TypeError(f"Unknown segmentation type {type(ann['segmentation'])}!")
            gt_mask_rles.append(rle)
            gt_planes.append(ann['plane'])
        
        gt_boxes = np.array(gt_boxes)  # xywh from coco
        gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        faux_gt_targets = Boxes(torch.tensor(gt_boxes, dtype=torch.float32, device=device))

        # box iou
        boxiou = pairwise_iou(boxes, faux_gt_targets)

        # filter predictions with iou > filter_iou
        # valid_pred_ids = (boxiou > filter_iou).sum(axis=1) > 0

        # mask iou
        miou = mask_util.iou(masks_rles, gt_mask_rles, [0]*len(gt_mask_rles))

        plane_metrics = compare_planes(planes, gt_planes)

        # sort predictions in descending order
        scores = torch.tensor(np.array(scores), dtype=torch.float32)
        scores_sorted, idx_sorted = torch.sort(scores, descending=True)
        # record assigned gt.
        box_covered = []
        mask_covered = []
        plane_covered = []


        for pred_id in range(num_img_preds):
            # remember we only evaluate the preds that have overlap more than
            # iou_filter with the ground truth prediction
            # if valid_pred_ids[idx_sorted[pred_id]] == 0:
            #     continue
            # Assign pred to gt
            gt_id = torch.argmax(boxiou[idx_sorted[pred_id]])
            gt_label = gt_labels[gt_id]
            # map to dataset category id
            pred_label = reverse_id_mapping[labels[idx_sorted[pred_id]]]
            pred_miou = miou[idx_sorted[pred_id], gt_id]
            pred_biou = boxiou[idx_sorted[pred_id], gt_id]
            pred_score = scores[idx_sorted[pred_id]].view(1).to(device)
            
            normal = plane_metrics['norm'][idx_sorted[pred_id], gt_id].item()
            offset = plane_metrics['offset'][idx_sorted[pred_id], gt_id].item()
            plane_offset_errs.append(offset)
            plane_normal_errs.append(normal)

            # mask
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_miou > iou_thresh)
                and (gt_id not in mask_covered)
            ):
                tpfp[0] = 1
                mask_covered.append(gt_id)
            mask_apscores[pred_label].append(pred_score)
            mask_aplabels[pred_label].append(tpfp)

            # box
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_biou > iou_thresh)
                and (gt_id not in box_covered)
            ):
                tpfp[0] = 1
                box_covered.append(gt_id)
            box_apscores[pred_label].append(pred_score)
            box_aplabels[pred_label].append(tpfp)

            # plane
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (normal < normal_threshold)
                and (offset < offset_threshold)
                and (gt_id not in plane_covered)
            ):
                tpfp[0] = 1
                plane_covered.append(gt_id)
            plane_apscores[pred_label].append(pred_score)
            plane_aplabels[pred_label].append(tpfp)

    # check things for eval
    # assert npos.sum() == len(dataset.dataset["annotations"])
    # convert to tensors
    detection_metrics = {}
    boxap, maskap, planeap = 0.0, 0.0, 0.0
    valid = 0.0
    for cat_id in cat_ids:
        cat_name = dataset.loadCats([cat_id])[0]["name"]
        if npos[cat_id] == 0:
            continue
        valid += 1

        cat_box_ap = VOCap.compute_ap(
            torch.cat(box_apscores[cat_id]), torch.cat(box_aplabels[cat_id]), npos[cat_id]
        ).item()
        boxap += cat_box_ap
        detection_metrics["box_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_box_ap

        cat_mask_ap = VOCap.compute_ap(
            torch.cat(mask_apscores[cat_id]), torch.cat(mask_aplabels[cat_id]), npos[cat_id]
        ).item()
        maskap += cat_mask_ap
        detection_metrics["mask_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_mask_ap

        cat_plane_ap = VOCap.compute_ap(
            torch.cat(plane_apscores[cat_id]), torch.cat(plane_aplabels[cat_id]), npos[cat_id]
        ).item()
        planeap += cat_plane_ap
        detection_metrics["plane_ap@iou%.1fnormal%.1foffset%.1f - %s" % (iou_thresh, normal_threshold, offset_threshold, cat_name)] = cat_plane_ap
    detection_metrics["box_ap@%.1f" % iou_thresh] = boxap / valid
    detection_metrics["mask_ap@%.1f" % iou_thresh] = maskap / valid
    detection_metrics["plane_ap@iou%.1fnormal%.1foffset%.1f" % (iou_thresh, normal_threshold, offset_threshold)] = planeap / valid
    logger.info("Detection metrics: \n"+create_small_table(detection_metrics))  
    plane_metrics = {}
    plane_normal_errs = np.array(plane_normal_errs)
    plane_offset_errs = np.array(plane_offset_errs)
    plane_metrics["%normal<10"] = sum(plane_normal_errs<10)/len(plane_normal_errs)*100
    plane_metrics["%normal<30"] = sum(plane_normal_errs<30)/len(plane_normal_errs)*100
    plane_metrics["%offset<0.5"] = sum(plane_offset_errs<0.5)/len(plane_offset_errs)*100
    plane_metrics["%offset<0.3"] = sum(plane_offset_errs<0.3)/len(plane_offset_errs)*100
    plane_metrics["mean_normal"] = plane_normal_errs.mean()
    plane_metrics["median_normal"] = np.median(plane_normal_errs)
    plane_metrics["mean_offset"] = plane_offset_errs.mean()
    plane_metrics["median_offset"] = np.median(plane_offset_errs)
    logger.info("Plane metrics: \n"+create_small_table(plane_metrics))  
    plane_metrics.update(detection_metrics)
    return plane_metrics



        
