from typing import Dict, List
import torch
from torch import nn

from detectron2.layers import ShapeSpec, cat

from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN


__all__ = ["DRPN"]

@PROPOSAL_GENERATOR_REGISTRY.register()
class DRPN(RPN):
    """
    Region Proposal Network with deeper conv layers.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        in_channels = 256
        self.rpn_head.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )
