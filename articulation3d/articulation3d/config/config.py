# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def get_planercnn_cfg_defaults(cfg):
    """
    Customize the detectron2 cfg to include some new keys and default values
    for Plane R-CNN
    """
    cfg.MODEL.FREEZE = []
    cfg.MODEL.PLANE_ON = True
    cfg.MODEL.DEPTH_ON = False
    cfg.MODEL.REFINE_ON = False
    cfg.MODEL.AXIS_ON = False
    cfg.MODEL.VIS_MINIBATCH = False  # visualize minibatches
    cfg.INPUT.IMG_HEIGHT = 480
    cfg.INPUT.IMG_WIDTH = 640
    cfg.INPUT.IMG_ROOT = '/z/syqian/articulation_data'

    # aspect ratio grouping has no difference in performance
    # but might reduce memory by a little bit
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    cfg.TEST.SAVE_VIS = False
    cfg.TEST.EVAL_GT_BOX = False

    # Foreground score threshold during test time
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    # ------------------------------------------------------------------------ #
    # Plane Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_PLANE_HEAD = CN()
    cfg.MODEL.ROI_PLANE_HEAD.NAME = "PlaneRCNNConvFCHead"
    cfg.MODEL.ROI_PLANE_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_PLANE_HEAD.POOLER_SAMPLING_RATIO = 0
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_PLANE_HEAD.POOLER_TYPE = "ROIAlign"
    cfg.MODEL.ROI_PLANE_HEAD.EMBEDDING_DIM = 128
    cfg.MODEL.ROI_PLANE_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_PLANE_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_PLANE_HEAD.NUM_FC = 1
    cfg.MODEL.ROI_PLANE_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_PLANE_HEAD.NORM = ""
    cfg.MODEL.ROI_PLANE_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_PLANE_HEAD.PARAM_DIM = 3
    cfg.MODEL.ROI_PLANE_HEAD.NORMAL_ONLY = True

    # ------------------------------------------------------------------------ #
    # Depth Predict Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.DEPTH_HEAD = CN()
    cfg.MODEL.DEPTH_HEAD.NAME = "PlaneRCNNDepthHead"
    cfg.MODEL.DEPTH_HEAD.LOSS_WEIGHT = 1.0
    # ------------------------------------------------------------------------ #
    # Mask Refine Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.REFINE_HEAD = CN()
    cfg.MODEL.REFINE_HEAD.NAME = "PlaneRCNNRefineHead"
    cfg.MODEL.REFINE_HEAD.LOSS_WEIGHT = 1.0
    # ------------------------------------------------------------------------ #
    # Mask Predict Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_MASK_HEAD.MASK_THRESHOLD = 0.5
    cfg.MODEL.ROI_MASK_HEAD.NMS = False
    # ------------------------------------------------------------------------ #
    # Plane Axis Prediction Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_AXIS_HEAD = CN()
    cfg.MODEL.ROI_AXIS_HEAD.NAME = "PlaneRCNNConvFCHead"
    cfg.MODEL.ROI_AXIS_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_AXIS_HEAD.POOLER_SAMPLING_RATIO = 0
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_AXIS_HEAD.POOLER_TYPE = "ROIAlign"
    cfg.MODEL.ROI_AXIS_HEAD.EMBEDDING_DIM = 128
    cfg.MODEL.ROI_AXIS_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_AXIS_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_AXIS_HEAD.NUM_FC = 1
    cfg.MODEL.ROI_AXIS_HEAD.FC_DIM = 1024 
    cfg.MODEL.ROI_AXIS_HEAD.PARAM_DIM = 3
    cfg.MODEL.ROI_AXIS_HEAD.NORM = ""
    cfg.MODEL.ROI_AXIS_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_AXIS_HEAD.SMOOTH_L1_BETA = 0.0
    return cfg
