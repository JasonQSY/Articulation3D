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
from packaging import version

import detectron2
import detectron2.utils.comm as comm
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou, PolygonMasks
from detectron2.utils.logger import setup_logger, create_small_table
from pycocotools.coco import COCO
from fvcore.common.file_io import PathManager, file_lock
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from articulation3d.evaluation.detectron2coco import convert_to_coco_dict
import articulation3d.utils.VOCap as VOCap
from articulation3d.utils.metrics import compare_axis, Line, EA_metric
from articulation3d.data.planercnn_transforms import axis_to_angle_offset, angle_offset_to_axis


logger = logging.getLogger(__name__)
if not logger.isEnabledFor(logging.INFO):
    setup_logger(name=__name__)


class ArtiEvaluator(COCOEvaluator):
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
        self._use_fast_impl = True
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

        self._tasks = {"bbox"}

        if 'SAVE_VIS' not in cfg.TEST:
            self._visualize = False
        else:
            self._visualize = cfg.TEST.SAVE_VIS
        
        if 'EVAL_GT_BOX' not in cfg.TEST:
            self._eval_gt_box = False
        else:
            self._eval_gt_box = cfg.TEST.EVAL_GT_BOX

        self._K_inv_dot_xy_1 = torch.FloatTensor(self.get_K_inv_dot_xy_1()).to(self._device)
        
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

    def override_offset(self, xyz, instance_coco, instance_d2):
        plane_normals = F.normalize(instance_d2['instances'].pred_plane, p=2)
        masks = instance_d2['instances'].pred_masks
        offsets = ((plane_normals.view(-1, 3, 1, 1) * xyz).sum(1) * masks).sum(-1).sum(-1) / torch.clamp(masks.sum(-1).sum(-1), min=1e-4)
        plane_parameters = plane_normals * offsets.view((-1, 1))
        valid = (masks.sum(-1).sum(-1) > 0).cpu()
        instance_coco['pred_plane'][valid] = plane_parameters.cpu()[valid]
        return instance_coco

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
        save_json = os.path.join(self._output_dir, 'arti_coco_'+d2json.replace('/', '_'))
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
                if hasattr(output['instances'], 'pred_rot_axis'):
                    prediction['pred_rot_axis'] = output['instances'].pred_rot_axis.to(self._cpu_device)
                if hasattr(output['instances'], 'pred_tran_axis'):
                    prediction['pred_tran_axis'] = output['instances'].pred_tran_axis.to(self._cpu_device)
                if hasattr(output['instances'], 'pred_plane'):
                    prediction['pred_plane'] = output['instances'].pred_plane.to(self._cpu_device)           
            if output['depth'] is not None:
                prediction['pred_depth'] = output['depth'].to(self._cpu_device)
            #        xyz = self.depth2XYZ(output['depth'])
            #        prediction = self.override_offset(xyz, prediction, output)
            self._predictions.append(prediction)

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

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            #gt_corrs = self._gt_corrs

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        print("debug")
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_recognition(predictions)
            self._eval_arti(predictions)

            
            #self._eval_predictions({"bbox"}, predictions)
            
            if version.parse(detectron2.__version__) >= version.parse('0.4'):
                self._eval_predictions(predictions)
            else:
                self._eval_predictions({"bbox"}, predictions)
            

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    
    def _eval_recognition(self, predictions):
        results = evaluate_for_recognition(
            predictions,
            self._coco_api,
            self._metadata,
            self._filter_iou,
            device=self._device,
        )
        self._results.update(results)


    def _eval_arti(self, predictions):
        if 'axis' not in predictions[0].keys():
            if 'pred_rot_axis' not in predictions[0].keys():
                if 'pred_tran_axis' not in predictions[0].keys():
                    return
        results = evaluate_for_arti_axis(
            predictions,
            self._coco_api,
            self._metadata,
            self._filter_iou,
            device=self._device,
        )
        self._results.update(results)


def evaluate_for_arti_axis(predictions, dataset, metadata, filter_iou, iou_thresh=0.5, normal_threshold=30, offset_threshold=100, device=None):
    if device is None:
        device = torch.device("cpu")
    # classes
    cat_ids = sorted(dataset.getCatIds())
    reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}

    # initialize tensors to record box & mask AP, number of gt positives
    box_apscores, box_aplabels, axis_box_apscores = {}, {}, {}
    axis_apscores, axis_aplabels, axis_box_aplabels = {}, {}, {}
    ap_scores = {
        'bbox': {},
        'bbox+axis': {},
        'bbox+normal': {},
        'bbox+normal+axis': {}, 
    }
    ap_labels = {
        'bbox': {},
        'bbox+axis': {},
        'bbox+normal': {},
        'bbox+normal+axis': {}, 
    }
    rot_EA = []
    tran_EA = []
    normal_errors = []
    npos = {}

    # initialize list per category
    for cat_id in cat_ids:
        for metric in ap_scores:
            ap_scores[metric][cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
            ap_labels[metric][cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]

        """
        box_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        box_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        axis_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        axis_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        axis_box_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        axis_box_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        """
        npos[cat_id] = 0.0

    # number of gt positive instances per class
    for gt_ann in dataset.dataset["annotations"]:
        gt_label = gt_ann["category_id"]
        npos[gt_label] += 1.0

    for prediction in predictions:
        original_id = prediction['image_id']
        if "instances" not in prediction:
            continue

        num_img_preds = len(prediction["instances"])
        if num_img_preds == 0:
            continue

        # predictions
        scores, boxes, labels = [], [], []
        for ins in prediction['instances']:
            scores.append(ins['score'])
            boxes.append(ins['bbox'])
            labels.append(ins['category_id'])
        boxes = np.array(boxes)  # xywh from coco
        boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        boxes = Boxes(torch.tensor(np.array(boxes))).to(device)
        axis_rot = prediction['pred_rot_axis']
        axis_tran = prediction['pred_tran_axis']
        try:
            pred_normals = F.normalize(prediction['pred_plane'], p=2)
        except:
            pred_normals = F.normalize(torch.FloatTensor(
                np.ones(
                    (len(scores), 3)
                )
            ), p=2)

        # convert pred normals from ScanNet/mp3d/SparsePlanes to SunCG
        pred_normals[:, [1,2]] = pred_normals[:, [2, 1]]
        pred_normals[:, 1] = - pred_normals[:, 1]

        # ground truth
        # anotations corresponding to original_id (aka coco image_id)
        gt_ann_ids = dataset.getAnnIds(imgIds=[original_id])
        gt_anns = dataset.loadAnns(gt_ann_ids)
        # get original ground truth mask, box, label & mesh
        gt_boxes, gt_labels, gt_rot_axis, gt_tran_axis, gt_normals = [], [], [], [], []
        for ann in gt_anns:
            gt_boxes.append(ann['bbox'])
            gt_labels.append(ann['category_id'])
            gt_rot_axis.append(ann['rot_axis'])
            gt_tran_axis.append(ann['tran_axis'])
            if 'normal' not in ann.keys() or ann['normal'] is None:
                gt_normals.append([-1, -1, -1])
            else:
                gt_normals.append(ann['normal'])
        if len(gt_boxes) == 0:
            continue
        gt_boxes = np.array(gt_boxes)  # xywh from coco
        gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        faux_gt_targets = Boxes(torch.tensor(gt_boxes, dtype=torch.float32, device=device))

        # gt normals come from OpenSurfaces, convert it to SUNCG
        gt_normals = torch.FloatTensor(gt_normals)
        #gt_normals[:, 0] = - gt_normals[:, 0]
        gt_normals[:, 1] = - gt_normals[:, 1]
        

        """
        get gt coord
        """
        gt_rot_ao = axis_to_angle_offset(gt_rot_axis, Boxes(gt_boxes).get_centers())
        gt_tran_ao = axis_to_angle_offset(gt_tran_axis, Boxes(gt_boxes).get_centers())

        valid_gt_rot = gt_rot_ao[:,3].ge(0.5)
        valid_gt_tran = gt_tran_ao[:,3].ge(0.5)

        try:
            gt_rot_coord = angle_offset_to_axis(gt_rot_ao[:,:3], Boxes(gt_boxes).get_centers())
        except:
            import pdb; pdb.set_trace()
            pass
        gt_tran_ao[:,2] = 0
        gt_tran_coord = angle_offset_to_axis(gt_tran_ao[:,:3], Boxes(gt_boxes).get_centers())

        """
        get pred coord
        """
        try:
            pred_rot_coord = angle_offset_to_axis(axis_rot, boxes.get_centers().cpu())
        except:
            import pdb; pdb.set_trace()
            pass
        tmp=torch.zeros(len(axis_tran),1)
        axis_tran = torch.cat((axis_tran, tmp), 1)
        pred_tran_coord = angle_offset_to_axis(axis_tran, boxes.get_centers().cpu())

        """
        EA_metrics
        """
        axis_rot_metrics = np.zeros((len(pred_rot_coord), len(gt_rot_coord)))
        for p in range(len(pred_rot_coord)):
            for g in range(len(gt_rot_coord)):
                p_coord = pred_rot_coord[p].tolist()
                if p_coord[0]==p_coord[2] and p_coord[1]==p_coord[3]:
                    axis_rot_metrics[p][g] = 0
                    continue
                l_pred = Line([p_coord[1], p_coord[0], p_coord[3], p_coord[2]])                
                g_coord = gt_rot_coord[g].tolist()
                try:
                    l_gt = Line([g_coord[1], g_coord[0], g_coord[3], g_coord[2]])
                except:
                    import pdb; pdb.set_trace()
                    pass
                axis_rot_metrics[p][g] = EA_metric(l_pred, l_gt)
        axis_tran_metrics = np.zeros((len(pred_tran_coord), len(gt_tran_coord)))
        for p in range(len(pred_tran_coord)):
            for g in range(len(gt_tran_coord)):
                p_coord = pred_tran_coord[p].tolist()
                if p_coord[0]==p_coord[2] and p_coord[1]==p_coord[3]:
                    axis_rot_metrics[p][g] = 0
                    continue
                l_pred = Line([p_coord[1], p_coord[0], p_coord[3], p_coord[2]])
                g_coord = gt_tran_coord[g].tolist()
                l_gt = Line([g_coord[1], g_coord[0], g_coord[3], g_coord[2]])
                axis_tran_metrics[p][g] = EA_metric(l_pred, l_gt)
        # import pdb;pdb.set_trace()
        
        
        

        # Compute box iou
        boxiou = pairwise_iou(boxes, faux_gt_targets)

        # Filter predictions with iou > filter_iou
        # Sort predictions in descending order
        valid_pred_ids = boxiou > filter_iou
        scores = torch.tensor(np.array(scores), dtype=torch.float32)
        scores_sorted, idx_sorted = torch.sort(scores, descending=True)
        
        # Record assigned gt.
        box_covered = {
            'bbox': [],
            'bbox+axis': [],
            'bbox+normal': [],
            'bbox+normal+axis': [],
        }
        #box_covered = []
        #axis_covered = []
        #axis_box_covered = []

        for pred_id in range(num_img_preds):
            # remember we only evaluate the preds that have overlap more than
            # iou_filter with the ground truth prediction
            if valid_pred_ids[idx_sorted[pred_id]] == 0:
                continue
            # Assign pred to gt
            gt_id = torch.argmax(boxiou[idx_sorted[pred_id]])
            gt_label = gt_labels[gt_id]
            # map to dataset category id
            pred_label = reverse_id_mapping[labels[idx_sorted[pred_id]]]
            pred_biou = boxiou[idx_sorted[pred_id], gt_id]
            pred_score = scores[idx_sorted[pred_id]].view(1).to(device)
            
            if 'rot' in metadata.thing_classes[metadata.thing_dataset_id_to_contiguous_id[gt_label]]:
                if valid_gt_rot[gt_id]:
                    pred_ea = axis_rot_metrics[idx_sorted[pred_id], gt_id].item()
                else:
                    pred_ea = 0
            elif 'tran' in metadata.thing_classes[metadata.thing_dataset_id_to_contiguous_id[gt_label]]:
                if valid_gt_tran[gt_id]:
                    pred_ea = axis_tran_metrics[idx_sorted[pred_id], gt_id].item()
                else:
                    pred_ea = 0
            else:
                raise NotImplementedError

            normal_error = torch.acos(torch.dot(pred_normals[pred_id], gt_normals[gt_id]))
            normal_error = normal_error / np.pi * 180.0
            if torch.norm(gt_normals[gt_id]) > 1.1: # invalid gt (-1, -1, -1)
                normal_error = 180.0

            for metric in ('bbox', 'bbox+axis', 'bbox+normal', 'bbox+normal+axis'):
                tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
                
                # bbox tp
                is_tp = (pred_label == gt_label) and (pred_biou > iou_thresh) and (gt_id not in box_covered[metric])
                
                # add additional tp constraints
                if metric == 'bbox+axis':
                    is_tp = is_tp and (pred_ea > iou_thresh)
                elif metric == 'bbox+normal':
                    is_tp = is_tp and (normal_error < normal_threshold)
                elif metric == 'bbox+normal+axis':
                    is_tp = is_tp and (pred_ea > iou_thresh) and (normal_error < normal_threshold)
                
                if is_tp:
                    tpfp[0] = 1
                    box_covered[metric].append(gt_id)

                ap_scores[metric][pred_label].append(pred_score)
                ap_labels[metric][pred_label].append(tpfp)

            # box
            """
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_biou > iou_thresh)
                and (gt_id not in box_covered)
            ):
                tpfp[0] = 1
                box_covered.append(gt_id)
                if 'rot' in metadata.thing_classes[metadata.thing_dataset_id_to_contiguous_id[gt_label]]:
                    if valid_gt_rot[gt_id]:
                        rot_EA.append(pred_ea)
                elif 'tran' in metadata.thing_classes[metadata.thing_dataset_id_to_contiguous_id[gt_label]]:
                    if valid_gt_tran[gt_id]:
                        tran_EA.append(pred_ea)
                else:
                    raise NotImplementedError

                #normal_error = torch.acos(torch.dot(pred_normals[pred_id], gt_normals[gt_id]))
                #if torch.norm(gt_normals) < 1.1: # exclude invalid gt
                #    normal_errors.append(normal_error)

            box_apscores[pred_label].append(pred_score)
            box_aplabels[pred_label].append(tpfp)

            # axis
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_ea > iou_thresh)
                and (gt_id not in axis_covered)
            ):
                tpfp[0] = 1
                axis_covered.append(gt_id)
            axis_apscores[pred_label].append(pred_score)
            axis_aplabels[pred_label].append(tpfp)

            # box + axis
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_ea > iou_thresh)
                and (pred_biou > iou_thresh)
                and (gt_id not in axis_box_covered)
            ):
                tpfp[0] = 1
                axis_box_covered.append(gt_id)
            axis_box_apscores[pred_label].append(pred_score)
            axis_box_aplabels[pred_label].append(tpfp)

            # box + normal
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_biou > iou_thresh)
                and (normal_error < normal_threshold)
                and (gt_id not in axis_box_covered)
            ):
                tpfp[0] = 1
                axis_box_covered.append(gt_id)
            normal_box_apscores[pred_label].append(pred_score)
            normal_box_aplabels[pred_label].append(tpfp)

            # box + axis + normal
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_ea > iou_thresh)
                and (pred_biou > iou_thresh)
                and (gt_id not in axis_box_covered)
            ):
                tpfp[0] = 1
                axis_box_covered.append(gt_id)
            normal_box_apscores[pred_label].append(pred_score)
            normal_box_aplabels[pred_label].append(tpfp)
            """


    # check things for eval
    # assert npos.sum() == len(dataset.dataset["annotations"])
    # convert to tensors
    detection_metrics = {}
    boxap_metrics, axisap_metrics, allap_metrics = {}, {}, {}
    boxap, axisap, axisboxap = 0.0, 0.0, 0.0
    valid = 0.0
    for cat_id in cat_ids:
        cat_name = dataset.loadCats([cat_id])[0]["name"]
        if npos[cat_id] == 0:
            continue
        valid += 1

        for metric in ('bbox', 'bbox+axis', 'bbox+normal', 'bbox+normal+axis'):
            detection_metrics['{} - {}'.format(metric, cat_name)] = VOCap.compute_ap(
                torch.cat(ap_scores[metric][cat_id]), torch.cat(ap_labels[metric][cat_id]), npos[cat_id]
            )



        """
        try:
            cat_box_ap = VOCap.compute_ap(
                torch.cat(box_apscores[cat_id]), torch.cat(box_aplabels[cat_id]), npos[cat_id]
            ).item()
        except:
            cat_box_ap = VOCap.compute_ap(
                torch.cat(box_apscores[cat_id]), torch.cat(box_aplabels[cat_id]), npos[cat_id]
            )
        boxap += cat_box_ap
        boxap_metrics["box_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_box_ap * 100

        cat_axis_ap = VOCap.compute_ap(
            torch.cat(axis_apscores[cat_id]), torch.cat(axis_aplabels[cat_id]), npos[cat_id]
        )
        axisap += cat_axis_ap
        axisap_metrics["axis_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_axis_ap * 100
    
        cat_axisbox_ap = VOCap.compute_ap(
            torch.cat(axis_box_apscores[cat_id]), torch.cat(axis_box_aplabels[cat_id]), npos[cat_id]
        )
        axisboxap += cat_axisbox_ap
        allap_metrics["axis_box_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_axisbox_ap * 100
        """

    logger.info("Detection metrics: \n"+create_small_table(detection_metrics))      

    """
    boxap_metrics["box_ap@%.1f" % iou_thresh] = boxap / valid * 100
    axisap_metrics["axis_ap@%.1f" % iou_thresh] = axisap / valid * 100
    allap_metrics["axis_box_ap@%.1f" % iou_thresh] = axisboxap / valid * 100
    # detection_metrics["axis_ap@iou%.1fnormal%.1foffset%.1f" % (iou_thresh, normal_threshold, offset_threshold)] = axisap / valid
    logger.info("Detection metrics (box): \n"+create_small_table(boxap_metrics))  
    logger.info("Detection metrics (axis): \n"+create_small_table(axisap_metrics))  
    logger.info("Detection metrics (all): \n"+create_small_table(allap_metrics))
    detection_metrics.update(boxap_metrics)
    detection_metrics.update(axisap_metrics)
    detection_metrics.update(allap_metrics)
    """

    """
    rot_axis_metrics = {}
    rot_EA = np.array(rot_EA)
    rot_axis_metrics["mean_rot_EA"] = rot_EA.mean()
    rot_axis_metrics["med_rot_EA"] = np.median(rot_EA)
    logger.info("Rotation Axis metrics: \n"+create_small_table(rot_axis_metrics))
    detection_metrics.update(rot_axis_metrics)

    tran_axis_metrics = {}
    tran_EA = np.array(tran_EA)
    tran_axis_metrics["avg_tran_EA"] = tran_EA.mean()
    tran_axis_metrics["med_tran_EA"] = np.median(tran_EA)    
    logger.info("Translation Axis metrics: \n"+create_small_table(tran_axis_metrics)) 
    detection_metrics.update(tran_axis_metrics)
    """

    #normal_errors = np.array(normal_errors)
    # normal_errors = normal_errors[~np.isnan(normal_errors)]
    #logger.info("normal_error: {}".format(normal_errors.mean()))
    #import pdb; pdb.set_trace()
     
    return detection_metrics



def evaluate_for_recognition(predictions, dataset, metadata, filter_iou, iou_thresh=0.5, normal_threshold=30, offset_threshold=100, device=None):
    if device is None:
        device = torch.device("cpu")
    # classes
    cat_ids = sorted(dataset.getCatIds())
    reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}

    # initialize tensors to record box & mask AP, number of gt positives
    box_apscores, box_aplabels = {}, {}
    axis_apscores, axis_aplabels = {}, {}
    axis_offset_errs, axis_normal_errs = [], []
    npos = {}
    for cat_id in cat_ids:
        box_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        box_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        axis_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        axis_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        npos[cat_id] = 0.0

    # number of gt positive instances per class
    for gt_ann in dataset.dataset["annotations"]:
        gt_label = gt_ann["category_id"]
        npos[gt_label] += 1.0

    preds = []
    gts = []

    for prediction in predictions:
        original_id = prediction['image_id']
        image_width = dataset.loadImgs([original_id])[0]["width"]
        image_height = dataset.loadImgs([original_id])[0]["height"]
        #if "instances" not in prediction:
        #    continue

        #num_img_preds = len(prediction["instances"])
        #if num_img_preds == 0:
        #    continue

        # predictions
        scores, boxes, labels = [], [], []
        for ins in prediction['instances']:
            scores.append(ins['score'])
            boxes.append(ins['bbox'])
            labels.append(ins['category_id'])
        #boxes = np.array(boxes)  # xywh from coco
        #boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        #boxes = Boxes(torch.tensor(np.array(boxes))).to(device)
        # axis = prediction['pred_axis']

        # ground truth
        # anotations corresponding to original_id (aka coco image_id)
        gt_ann_ids = dataset.getAnnIds(imgIds=[original_id])
        gt_anns = dataset.loadAnns(gt_ann_ids)
        # get original ground truth mask, box, label & mesh
        gt_boxes, gt_labels = [], []
        for ann in gt_anns:
            gt_boxes.append(ann['bbox'])
            gt_labels.append(ann['category_id'])

        pred = 0
        if len(scores) > 0:
            pred = np.array(scores).max()
        gt = len(gt_boxes) > 0
        preds.append(pred)
        gts.append(gt)
       
    preds = np.array(preds)
    gts = np.array(gts)

    '''
    import pdb; pdb.set_trace()
    with open('preds.npy', 'wb') as f:
        np.save(f, preds)

    with open('gts.npy', 'wb') as f:
        np.save(f, gts)
    '''
    recog_metrics = {}
    try:
        recog_metrics['auroc'] = roc_auc_score(gts, preds)
        recog_metrics['accuracy'] = ((preds > 0.95) == gts).sum() / (preds == preds).sum()
    except:
        recog_metrics['auroc'] = -1
        recog_metrics['accuracy'] = -1

    logger.info("Recognition results: \n"+create_small_table(recog_metrics)) 

    return recog_metrics

