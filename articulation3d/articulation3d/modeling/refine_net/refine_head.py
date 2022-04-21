"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import numpy as np
from typing import Dict
from torch import nn
from torch.nn import functional as F
from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import polygons_to_bitmask

__all__ = ["build_refine_head", "PlaneRCNNRefineHead", "REFINE_HEAD_REGISTRY"]

REFINE_HEAD_REGISTRY = Registry("REFINE_HEAD")
REFINE_HEAD_REGISTRY.__doc__ = """
Registry for mask refine head in a generalized R-CNN model.
ROIHeads take feature maps and predict depth.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`nn.module`.
"""

def build_refine_head(cfg):
    """
    Build Refine head defined by `cfg.MODEL.REFINE_HEAD.NAME`.
    """
    name = cfg.MODEL.REFINE_HEAD.NAME
    return REFINE_HEAD_REGISTRY.get(name)(cfg)


class ConvBlock(torch.nn.Module):
    """The block consists of a convolution layer, an optional batch normalization layer, and a ReLU layer"""
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, output_padding=0, mode='conv', use_bn=True):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        if mode == 'conv':
            self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not self.use_bn)
        elif mode == 'deconv':
            self.conv = torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not self.use_bn)
        elif mode == 'upsample':
            self.conv = torch.nn.Sequential(torch.nn.Upsample(scale_factor=stride, mode='nearest'), torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=not self.use_bn))
        elif mode == 'conv_3d':
            self.conv = torch.nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not self.use_bn)
        elif mode == 'deconv_3d':
            self.conv = torch.nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not self.use_bn)
        else:
            print('conv mode not supported', mode)
            exit(1)
            pass
        if self.use_bn:
            if '3d' not in mode:
                self.bn = torch.nn.BatchNorm2d(out_planes)
            else:
                self.bn = torch.nn.BatchNorm3d(out_planes)
                pass
        self.relu = torch.nn.ReLU(inplace=True)
        return
   
    def forward(self, inp):
        if self.use_bn:
            return self.relu(self.bn(self.conv(inp)))
        else:
            return self.relu(self.conv(inp))


class RefinementBlockMask(torch.nn.Module):
    def __init__(self, options):
        super(RefinementBlockMask, self).__init__()
        self.options = options
        use_bn = False
        self.conv_0 = ConvBlock(3 + 5 + 1, 32, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
        self.conv_1 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1, use_bn=use_bn)       
        self.conv_1_1 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
        self.conv_2 = ConvBlock(128, 128, kernel_size=3, stride=2, padding=1, use_bn=use_bn)
        self.conv_2_1 = ConvBlock(256, 128, kernel_size=3, stride=1, padding=1, use_bn=use_bn)

        self.up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
        self.up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)              
        self.pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                    torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))

        self.global_up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
        self.global_up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)              
        self.global_pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                        torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))
        

    def accumulate(self, x):
        return torch.cat([x, (x.sum(0, keepdim=True) - x) / max(len(x) - 1, 1)], dim=1)
        
    def forward(self, image, masks):
        """
        image - torch.Size([#segm, 3, 192, 256])
        masks - torch.Size([#segm, 6, 192, 256])
        """
        # import pdb;pdb.set_trace()
        # self.vis_binary_mask(image / 255, 'input_image')
        # for i in range(len(masks)):
        #     self.vis_binary_mask(masks[i], f'input_masks_{i}')  
        x_0 = torch.cat([image, masks], dim=1)

        x_0 = self.conv_0(x_0)
        x_1 = self.conv_1(self.accumulate(x_0))
        x_1 = self.conv_1_1(self.accumulate(x_1))
        x_2 = self.conv_2(self.accumulate(x_1))
        x_2 = self.conv_2_1(self.accumulate(x_2))
        
        y_2 = self.up_2(x_2)
        y_1 = self.up_1(torch.cat([y_2, x_1], dim=1))
        y_0 = self.pred(torch.cat([y_1, x_0], dim=1))
        
        global_y_2 = self.global_up_2(x_2.mean(dim=0, keepdim=True))
        global_y_1 = self.global_up_1(torch.cat([global_y_2, x_1.mean(dim=0, keepdim=True)], dim=1))
        global_mask = self.global_pred(torch.cat([global_y_1, x_0.mean(dim=0, keepdim=True)], dim=1))

        y_0 = torch.cat([global_mask[:, 0], y_0.squeeze(1)], dim=0)
        return y_0

    
    def vis_binary_mask(self, mask, prefix, folder='debug'):
        import cv2, os
        for i in range(len(mask)):
            if mask[i].dim() == 3:
                cv2.imwrite(os.path.join(folder, prefix+f'_c{i}.png'),  (mask[i]*255).cpu().numpy()[0])
            else:
                cv2.imwrite(os.path.join(folder, prefix+f'_c{i}.png'),  (mask[i]*255).cpu().numpy())



"""
RefinementNet, refine depth output
"""
class RefinementNet(nn.Module):

    def __init__(self, options, resize):
        super(RefinementNet, self).__init__()
        self.options = options
        self.refinement_block = RefinementBlockMask(options)
        self.resize = resize

    def forward(self, image, predictions):
        masks_inp = torch.cat([predictions['mask'].unsqueeze(1), predictions['XYZ_plane']], dim=1)
        prev_predictions = torch.cat([
            predictions['raw_depth'].repeat((len(masks_inp), 1, 1, 1)),  # 1 channel
            masks_inp,  # 4
            (masks_inp.sum(0, keepdim=True) - masks_inp)[:, :1] # 1
            ], dim=1)
        
        masks = self.refinement_block(image.repeat((len(masks_inp), 1, 1, 1)), prev_predictions)
        return masks


@REFINE_HEAD_REGISTRY.register()
class PlaneRCNNRefineHead(nn.Module):
    def __init__(self, options):
        super(PlaneRCNNRefineHead, self).__init__()
        self.options = options
        # resize to reduce GPU memory
        self.resize = True
        self.refinement_net = RefinementNet(options, self.resize)
        self.ranges = self.get_ranges()
        self.ranges.requires_grad=False
        
    
    def get_ranges(self, h=480, w=640, focal_length=571.623718):
        with torch.no_grad():
            urange_unit = ((torch.arange(w, requires_grad=False).cuda().float() + 0.5) / w).view((1, -1)).repeat(h, 1)
            vrange_unit = ((torch.arange(h, requires_grad=False).cuda().float() + 0.5) / h).view((-1, 1)).repeat(1, w)
            ones = torch.ones(urange_unit.shape, requires_grad=False).cuda()
        urange = (urange_unit * w - w / 2) / focal_length
        vrange = (vrange_unit * h - h / 2) / focal_length
        ranges = torch.stack([urange, ones, -vrange], dim=-1)

        return ranges

    def planeXYZModule(self, planes, width=480, height=640, max_depth=10):
        """Compute plane XYZ from plane parameters
        ranges: K^(-1)x
        planes: plane parameters

        Returns:
        plane depthmaps
        """
        planeOffsets = torch.norm(planes, dim=-1, keepdim=True)
        planeNormals = planes / torch.clamp(planeOffsets, min=1e-4)

        normalXYZ = torch.matmul(self.ranges, planeNormals.transpose(0, 1))
        normalXYZ[normalXYZ == 0] = 1e-4
        planeDepths = planeOffsets.squeeze(-1) / normalXYZ
        planeDepths = torch.clamp(planeDepths, min=0, max=max_depth)
        planeXYZ = planeDepths.unsqueeze(-1) * self.ranges.unsqueeze(2)
        return planeXYZ.transpose(2, 3).transpose(1, 2).transpose(0, 1).transpose(2, 3).transpose(1, 2).transpose(0, 1)

    def assign_pred_mask_with_gt_mask(self, gt_masks, pred_masks):
        """
        for each predicted mask, find gt mask with the highest intersection area.
        Input:
            gt_masks: polygon mask
            pred_masks: bitmask
        return:
            masks_gt: for each predicted masks, assign the gt mask with highest intersection
            valid_mask: whether the gt_masks has the highest intersection with the assigned pred mask
        """
        masks_gt = [polygons_to_bitmask(p, 480, 640) for p in gt_masks.polygons]
        masks_gt = torch.stack([torch.from_numpy(x) for x in masks_gt]).unsqueeze(1).to("cuda")
        intersection = (torch.round(pred_masks).long() * masks_gt).sum(-1).sum(-1)
        _, segments_gt = intersection.max(0)
        mapping = intersection.max(1)[1]
        valid_mask = (mapping[segments_gt] == torch.arange(len(segments_gt)).cuda()).float()
        masks_gt = masks_gt[segments_gt]
        return masks_gt, valid_mask

    
    def vis_binary_mask(self, mask, prefix, folder='debug'):
        import cv2, os
        for i in range(len(mask)):
            if mask[i].dim() == 3:
                cv2.imwrite(os.path.join(folder, prefix+f'_c{i}.png'),  (mask[i]*255).cpu().numpy()[0])
            else:
                cv2.imwrite(os.path.join(folder, prefix+f'_c{i}.png'),  (mask[i]*255).cpu().numpy())


    def loss(self, masks_gt, masks_pred, valid_mask):
        """
        Input: 
            masks_gt: binary masks # torch.Size([#mask, 1, 192, 256])
            masks_pred: logits # torch.Size([1+#mask, 192, 256])
        """
        # import pdb;pdb.set_trace()
        masks_gt = masks_gt.float()
        all_masks_gt = torch.cat([1 - masks_gt.max(dim=0, keepdim=True)[0], masks_gt], dim=0) # torch.Size([5, 1, 192, 256])
        segmentation = all_masks_gt.max(0)[1].view(-1) # torch.Size([49152])
        masks_logits = masks_pred.squeeze(1).transpose(0, 1).transpose(1, 2).contiguous().view((segmentation.shape[0], -1))  # torch.Size([49152, 5])
        mask_loss = torch.nn.functional.cross_entropy(masks_logits, segmentation, weight=torch.cat([torch.ones(1).cuda(), valid_mask], dim=0))
        return mask_loss

    
    def forward(self, batched_inputs, pred_instances, pred_depth):
        ## Compute only plane offset using depthmap prediction
        refine_loss = torch.zeros(1).cuda()
        for i in range(len(batched_inputs)):
            if len(pred_instances[i]['instances']) == 0:
                continue
            plane_normals = pred_instances[i]['instances'].pred_plane
            # print(f"# of plane {len(plane_normals)}")
            masks = pred_instances[i]['instances'].pred_masks / 255.0
            XYZ_np = self.ranges.transpose(1, 2).transpose(0, 1) * pred_depth[i].unsqueeze(0)
            offsets = ((plane_normals.view(-1, 3, 1, 1) * XYZ_np).sum(1) * masks).sum(-1).sum(-1) / torch.clamp(masks.sum(-1).sum(-1), min=1e-4)
            plane_parameters = plane_normals * offsets.view((-1, 1))
            XYZ_plane = self.planeXYZModule(plane_parameters)
            predictions = {'mask': masks, 'raw_depth': pred_depth[i].unsqueeze(0).unsqueeze(0), 'XYZ_plane': XYZ_plane}

            image = batched_inputs[i]['image'].to("cuda") / 255.0

            if self.resize:
                image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(192, 256), mode='bilinear').squeeze(0)
                for key in predictions.keys():
                    if predictions[key].dim() == 4:
                        predictions[key] = torch.nn.functional.interpolate(predictions[key], size=(192, 256), mode='bilinear')
                    else:
                        predictions[key] = torch.nn.functional.interpolate(predictions[key].unsqueeze(0), size=(192, 256), mode='bilinear').squeeze(0)
            
            refined_masks = self.refinement_net(image, predictions)
            if self.training:
                ## Generate supervision target for the refinement network
                masks_gt, valid_mask = self.assign_pred_mask_with_gt_mask(batched_inputs[i]['instances'].gt_masks, masks)
                if self.resize:
                    masks_gt = torch.nn.functional.interpolate(masks_gt.float(), size=(192, 256), mode='bilinear')
                # import pdb;pdb.set_trace()
                # self.vis_binary_mask(predictions['mask'], 'pred')
                # self.vis_binary_mask(masks_gt, 'gt')
                # self.vis_binary_mask(torch.softmax(refined_masks, dim=0).detach(), 'refined')
                # self.vis_binary_mask(XYZ_plane, 'XYZ_plane')
                # self.vis_binary_mask(predictions['raw_depth']/predictions['raw_depth'].max(), 'depth')

                refine_loss += self.loss(masks_gt, refined_masks, valid_mask)

            else:
                masks_pred = (refined_masks.max(0, keepdim=True)[1] == torch.arange(len(refined_masks)).cuda().long().view((-1, 1, 1))).float()[1:]
                if self.resize:
                    masks_pred = torch.nn.functional.interpolate(masks_pred.unsqueeze(1), size=(480, 640), mode='bilinear').squeeze(1)                        
                
                pred_instances[i]['instances'].pred_masks = masks_pred
                pred_instances[i]['instances'].pred_plane = plane_parameters
        if self.training:
            return {"refine_loss": refine_loss}
        else:
            return pred_instances
