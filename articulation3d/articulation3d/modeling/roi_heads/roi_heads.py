# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict
import torch
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss
from detectron2.structures import Boxes, Instances

from articulation3d.modeling.roi_heads.plane_head import (
    build_plane_head,
)
from articulation3d.modeling.roi_heads.axis_head import (
    build_axis_head,
)



@ROI_HEADS_REGISTRY.register()
class PlaneRCNNROIHeads(StandardROIHeads):
    """
    The ROI specific heads for Mesh R-CNN
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self._init_plane_head(cfg, input_shape)
        self._init_axis_head(cfg, input_shape)
        self._vis = cfg.MODEL.VIS_MINIBATCH
        self._misc = {}
        self._vis_dir = cfg.OUTPUT_DIR
        self._vis_period = cfg.VIS_PERIOD
        self._eval_gt_box = cfg.TEST.EVAL_GT_BOX
        self._refine_on = cfg.MODEL.REFINE_ON
        self._freeze = cfg.MODEL.FREEZE

    def _init_plane_head(self, cfg, input_shape):
        self.plane_on = cfg.MODEL.PLANE_ON
        if not self.plane_on:
            return

        plane_pooler_resolution = cfg.MODEL.ROI_PLANE_HEAD.POOLER_RESOLUTION
        plane_pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        plane_sampling_ratio    = cfg.MODEL.ROI_PLANE_HEAD.POOLER_SAMPLING_RATIO
        plane_pooler_type       = cfg.MODEL.ROI_PLANE_HEAD.POOLER_TYPE
        
        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.plane_pooler = ROIPooler(
            output_size=plane_pooler_resolution, 
            scales=plane_pooler_scales, 
            sampling_ratio=plane_sampling_ratio,
            pooler_type=plane_pooler_type,
        )        
        shape = ShapeSpec(
            channels=in_channels, width=plane_pooler_resolution, height=plane_pooler_resolution
        )
        self.plane_head = build_plane_head(cfg, shape)

    
    def _init_axis_head(self, cfg, input_shape):
        self.axis_on = cfg.MODEL.AXIS_ON
        if not self.axis_on:
            return

        axis_pooler_resolution = cfg.MODEL.ROI_AXIS_HEAD.POOLER_RESOLUTION
        axis_pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        axis_sampling_ratio    = cfg.MODEL.ROI_AXIS_HEAD.POOLER_SAMPLING_RATIO
        axis_pooler_type       = cfg.MODEL.ROI_AXIS_HEAD.POOLER_TYPE
        
        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.axis_pooler = ROIPooler(
            output_size=axis_pooler_resolution, 
            scales=axis_pooler_scales, 
            sampling_ratio=axis_sampling_ratio,
            pooler_type=axis_pooler_type,
        )        
        shape = ShapeSpec(
            channels=in_channels, width=axis_pooler_resolution, height=axis_pooler_resolution
        )
        self.axis_head = build_axis_head(cfg, shape)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        feature: feature from FPN
        """
        if self._vis:
            self._misc["images"] = images
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets

        if self._vis:
            self._misc["proposals"] = proposals
        if self.training:
            losses = {}
            box_losses, pred_instances = self._forward_box(features, proposals)
            roi_losses = self.forward_with_selected_boxes(features, proposals, targets)

            if self._refine_on:
                self.training = False
                self.mask_head.training = False
                self.plane_head.training = False
                with torch.no_grad():
                    pred_instances = self.forward_with_given_boxes(features, pred_instances)
                self.training = True
                self.mask_head.training = True
                self.plane_head.training = True

            losses.update(box_losses)
            losses.update(roi_losses)
            del targets
            return losses, pred_instances
        else:
            if self._eval_gt_box:
                pred_instances = [Instances(p._image_size) for p in proposals]
                for ins, p in zip(pred_instances, proposals):
                    ins.pred_boxes = p.gt_boxes
                    ins.scores = torch.ones(len(p.gt_boxes)).to("cuda")
                    ins.pred_classes = p.gt_classes
            else:
                pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_selected_boxes(self, features, instances, targets=None):
        assert self.training
        losses = {}
        pred_instances, _ = select_foreground_proposals(instances, self.num_classes)
        if 'roi_heads.mask_head' not in self._freeze:
            losses.update(self._forward_mask(features, pred_instances))
        if 'roi_heads.plane_head' not in self._freeze:
            plane_loss = self._forward_plane(features, pred_instances)
            losses.update(plane_loss)
        if 'roi_heads.axis_head' not in self._freeze:
            axis_loss = self._forward_axis(features, pred_instances)
            losses.update(axis_loss)
        return losses


    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances): the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_voxels`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        instances = self._forward_mask(features, instances)
        instances = self._forward_plane(features, instances)
        instances = self._forward_axis(features, instances)
        return instances

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            if self._refine_on:
                with torch.no_grad():
                    pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                return losses, pred_instances
            return losses, None
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str,Tensor]): mapping from names to backbone features
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals = instances
            # Use Pred box to train
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            mask_logits = self.mask_head.layers(mask_features)
            return {"loss_mask": mask_rcnn_loss(mask_logits, proposals, self._vis_period)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_plane(self, features, instances):
        if not self.plane_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals = instances
            # Use Pred box to train
            proposal_boxes = [x.proposal_boxes for x in proposals]            
        else:
            proposal_boxes = [x.pred_boxes for x in instances]
        plane_features = self.plane_pooler(features, proposal_boxes)
        
        if self.training:
            return self.plane_head(plane_features, proposals)
        else:
            return self.plane_head(plane_features, instances)

    def _forward_axis(self, features, instances):
        if not self.axis_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals = instances
            # Use Pred box to train
            proposal_boxes = [x.proposal_boxes for x in proposals]            
        else:
            proposal_boxes = [x.pred_boxes for x in instances]
        axis_features = self.axis_pooler(features, proposal_boxes)
        
        if self.training:
            return self.axis_head(axis_features, proposals)
        else:
            return self.axis_head(axis_features, instances)
