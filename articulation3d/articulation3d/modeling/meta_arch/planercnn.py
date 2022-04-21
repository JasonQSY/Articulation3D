import logging
import numpy as np
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.structures import Instances
from detectron2.modeling import (
    build_backbone,
    build_proposal_generator,
    build_roi_heads, 
)
from detectron2.modeling import META_ARCH_REGISTRY
from articulation3d.modeling.postprocessing import detector_postprocess

from articulation3d.modeling.depth_net import build_depth_head
from articulation3d.modeling.refine_net import build_refine_head


__all__ = ["PlaneRCNN"]

@META_ARCH_REGISTRY.register()
class PlaneRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.mask_threshold = cfg.MODEL.ROI_MASK_HEAD.MASK_THRESHOLD
        self.nms = cfg.MODEL.ROI_MASK_HEAD.NMS
        self.depth_head_on = cfg.MODEL.DEPTH_ON
        self.refine_on = cfg.MODEL.REFINE_ON
        self.axis_on = cfg.MODEL.AXIS_ON
        if self.depth_head_on:
            self.depth_head = build_depth_head(cfg)
        if self.refine_on:
            self.refine_head = build_refine_head(cfg)
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self._eval_gt_box = cfg.TEST.EVAL_GT_BOX
        self.to(self.device)
        self._freeze = cfg.MODEL.FREEZE
        for layers in self._freeze:
            layer = layers.split('.')
            final = self
            for l in layer:
                final = getattr(final, l)
            for params in final.parameters():
                params.requires_grad = False

        
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        # FIXME: if use gt box, put it here. If freeze proposal_generator, do not need losses.
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            if 'proposal_generator' in self._freeze:
                proposal_losses = {}
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        detector_losses, pred_instances = self.roi_heads(images, features, proposals, gt_instances)
        depth_losses = {}
        if self.depth_head_on and 'depth_head' not in self._freeze:
            gt_depth = self.process_depth(batched_inputs)
            pred_depth, depth_losses = self.depth_head(features, gt_depth)
        
        refine_losses = {}
        if self.refine_on:
            pred_instances = PlaneRCNN._postprocess(pred_instances, batched_inputs, images.image_sizes, mask_threshold=-1, nms=self.nms)
            refine_losses = self.refine_head(batched_inputs, pred_instances, pred_depth)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(depth_losses)
        losses.update(refine_losses)
        # import pdb;pdb.set_trace()
        # losses['refine_loss'].backward(torch.ones(losses['refine_loss'].shape).to("cuda"))
        # [param.grad for param in self.refine_head.refinement_net.refinement_block.parameters()]
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        assert detected_instances is None

        pred_instances, pred_depth = self.inference_single(batched_inputs, do_postprocess)
        for pre, d, in zip(pred_instances, pred_depth):
            pre.update({'depth': d})
        return pred_instances

    def inference_single(self, batched_inputs, do_postprocess=True):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        #import pdb; pdb.set_trace()
        if self._eval_gt_box:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            elif "targets" in batched_inputs[0]:
                log_first_n(
                    logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
                )
                gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            for inst in gt_instances:
                inst.proposal_boxes = inst.gt_boxes
                inst.objectness_logits = torch.ones(len(inst.gt_boxes))
            proposals = gt_instances
        else:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        pred_depth = [None]*len(proposals)
        if self.depth_head_on:
            pred_depth = self.depth_head(features, None)

        results, _ = self.roi_heads(images, features, proposals, None)

        if self.refine_on:
            pred_instances = PlaneRCNN._postprocess(results, batched_inputs, images.image_sizes, mask_threshold=-1, nms=self.nms)
            results = self.refine_head(batched_inputs, pred_instances, pred_depth)
            return results, pred_depth

        else:
            return PlaneRCNN._postprocess(results, batched_inputs, images.image_sizes, mask_threshold=self.mask_threshold, nms=self.nms), pred_depth



    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = {}
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def process_depth(self, batched_inputs):
        depth = [x["depth"].to(self.device) for x in batched_inputs]
        depth = torch.stack(depth)
        return depth

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes, mask_threshold=0.5, nms=False):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            #r = detector_postprocess(results_per_image, height, width, mask_threshold, nms=nms)
            #import pdb; pdb.set_trace()
            r = detector_postprocess(results_per_image, height, width, mask_threshold, box_score_threshold=0.1, nms=nms)
            processed_results.append({"instances": r})
        return processed_results
