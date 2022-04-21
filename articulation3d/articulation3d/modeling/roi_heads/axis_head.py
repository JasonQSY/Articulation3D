# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm
from detectron2.utils.registry import Registry
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

ROI_AXIS_HEAD_REGISTRY = Registry("ROI_AXIS_HEAD")


@ROI_AXIS_HEAD_REGISTRY.register()
class PlaneRCNNConvFCHead(nn.Module):
    """
    A head with several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_fc: the number of fc layers
            fc_dim: the dimension of the fc layers
        """
        super().__init__()

        # fmt: off
        num_conv        = cfg.MODEL.ROI_AXIS_HEAD.NUM_CONV
        conv_dim        = cfg.MODEL.ROI_AXIS_HEAD.CONV_DIM
        num_fc          = cfg.MODEL.ROI_AXIS_HEAD.NUM_FC
        fc_dim          = cfg.MODEL.ROI_AXIS_HEAD.FC_DIM
        param_dim       = cfg.MODEL.ROI_AXIS_HEAD.PARAM_DIM
        norm            = cfg.MODEL.ROI_AXIS_HEAD.NORM

        # fmt: on
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.smooth_l1_beta = cfg.MODEL.ROI_AXIS_HEAD.SMOOTH_L1_BETA
        print("smooth l1 beta: {}".format(self.smooth_l1_beta))

        self.conv_norm_relus_R = []
        self.conv_norm_relus_T = []
        for k in range(num_conv):
            conv_R = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm , conv_dim),
                activation=F.relu,
            )
            conv_T = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm , conv_dim),
                activation=F.relu,
            )
            self.add_module("axis_R_conv{}".format(k + 1), conv_R)
            self.conv_norm_relus_R.append(conv_R)
            self.add_module("axis_T_conv{}".format(k + 1), conv_T)
            self.conv_norm_relus_T.append(conv_T)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs_R = []
        self.fcs_T = []
        for k in range(num_fc):
            fc_R = nn.Linear(np.prod(self._output_size), fc_dim)
            fc_T = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("axis_R_fc{}".format(k + 1), fc_R)
            self.fcs_R.append(fc_R)
            self.add_module("axis_T_fc{}".format(k + 1), fc_T)
            self.fcs_T.append(fc_T)
            self._output_size = fc_dim

        self.rotation = nn.Linear(fc_dim, 2)
        self.offset = nn.Linear(fc_dim, 1)
        self.translation = nn.Linear(fc_dim, 2)

        for layer in self.conv_norm_relus_R:
            weight_init.c2_msra_fill(layer)
        for layer in self.conv_norm_relus_T:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs_R:
            weight_init.c2_xavier_fill(layer)
        for layer in self.fcs_T:
            weight_init.c2_xavier_fill(layer)

        self._loss_weight = cfg.MODEL.ROI_AXIS_HEAD.LOSS_WEIGHT

    def forward(self, x, instances):
        # Rotation Axis
        x_R = x
        for layer in self.conv_norm_relus_R:
            x_R = layer(x_R)
        if len(self.fcs_R):
            if x_R.dim() > 2:
                x_R = torch.flatten(x_R, start_dim=1)
            for layer in self.fcs_R:
                x_R = F.relu(layer(x_R))
        
        rotation = F.normalize(self.rotation(x_R), p=2, dim=1)
        offset = self.offset(x_R)
        pred_rot_axis = torch.cat((rotation, offset),dim=1)

        # Translation Axis
        x_T = x
        for layer in self.conv_norm_relus_T:
            x_T = layer(x_T)
        if len(self.fcs_T):
            if x_T.dim() > 2:
                x_T = torch.flatten(x_T, start_dim=1)
            for layer in self.fcs_T:
                x_T = F.relu(layer(x_T))

        pred_tran_axis = F.normalize(self.translation(x_T), p=2, dim=1)

        # Loss
        try:
            if self.training:
                return {"loss_rot_axis": axis_loss(pred_rot_axis, instances, loss_weight=self._loss_weight, smooth_l1_beta=self.smooth_l1_beta, axistype='rot'),
                        "loss_tran_axis": axis_loss(pred_tran_axis, instances, loss_weight=self._loss_weight, smooth_l1_beta=self.smooth_l1_beta, axistype='tran')}
            else:
                arti_inference(pred_rot_axis, pred_tran_axis, instances)
                return instances
        except:
            import pdb;pdb.set_trace()
            pass

    @property
    def output_size(self):
        return self._output_size


def double_angle(sin_cos):
    """ use double-angle formula to convert theta to 2*theta
    input [sin(alpha), cos(alpha)]
    output [sin(2alpha), cos(2alpha)]
    """
    sin = sin_cos[:,0].unsqueeze(1)
    cos = sin_cos[:,1].unsqueeze(1)
    sin2 = 2*sin*cos
    cos2 = cos**2 - sin**2
    return torch.cat((sin2,cos2), dim=1)


def axis_loss(pred_axis, instances, loss_weight=1.0, smooth_l1_beta=1.0, axistype='rot'):
    """
    Compute the plane_param loss.
    Args:
        z_pred (Tensor): A tensor of shape (B, C) or (B, 1) for class-specific or class-agnostic,
            where B is the total number of foreground regions in all images, C is the number of foreground classes,
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        loss (Tensor): A scalar tensor containing the loss.
    """
    gt_axis = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if axistype == 'rot': 
            gt_axis.append(instances_per_image.gt_rot_axis)
        elif axistype == 'tran':
            gt_axis.append(instances_per_image.gt_tran_axis)
        else:
            raise NotImplementedError()

    if len(gt_axis) == 0:
        return pred_axis.sum() * 0.

    gt_axis = cat(gt_axis, dim=0)
    assert len(gt_axis) > 0
    valid = gt_axis[:,3:4]

    if valid.sum() < 1:
        return pred_axis.sum() * 0.

    if axistype == 'rot': 
        loss_axis_reg = smooth_l1_loss(pred_axis, gt_axis[:,:3], smooth_l1_beta, reduction=None)
    elif axistype == 'tran':
        pred_axis_double = double_angle(pred_axis)
        gt_axis_double = double_angle(gt_axis[:,:2])
        loss_axis_reg = smooth_l1_loss(pred_axis_double, gt_axis_double, smooth_l1_beta, reduction=None)
    else:
        raise NotImplementedError()

    loss = loss_weight * torch.masked_select(loss_axis_reg,valid.ge(0.5)).mean()
    if torch.isnan(loss):
        print(pred_axis)
        print(gt_axis)
        print(loss_axis_reg)
        print(loss_weight)
        print(valid)
        raise loss
    return loss


def arti_inference(pred_rot_axis, pred_tran_axis, pred_instances):
    num_boxes_per_image = [len(i) for i in pred_instances]
    pred_rot_axis = pred_rot_axis.split(num_boxes_per_image, dim=0)
    pred_tran_axis = pred_tran_axis.split(num_boxes_per_image, dim=0)

    for r_axis, t_axis, instances in zip(pred_rot_axis, pred_tran_axis, pred_instances):
        instances.pred_rot_axis = r_axis
        instances.pred_tran_axis = t_axis


def build_axis_head(cfg, input_shape):
    name = cfg.MODEL.ROI_AXIS_HEAD.NAME
    return ROI_AXIS_HEAD_REGISTRY.get(name)(cfg, input_shape)