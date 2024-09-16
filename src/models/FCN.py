"""
FCN model.
"""

import torch.nn as nn
from typing import Any, Optional, Sequence
import torch.nn.functional as F
import torch

class FCNModel(nn.Module):
    def __init__(self, backbone):
        super(FCNModel, self).__init__()
        self.backbone = backbone.backbone
        #
        # Add reconstruction head
        #
        self.reconstruction_head = FCNHead(2048, 3)  # 960 for mobilenet
        #
        # activation function
        #
        self.activation = nn.Sigmoid()
        
    def forward(self, input):
        # extract features from backbone
        x = self.backbone(input)['out']
        # reconstruction
        x = self.reconstruction_head(x)
        x = F.interpolate(x, size=input.shape[-2:], mode="bilinear", align_corners=False)
        # activation
        x = self.activation(x)
        return x
    

"""
https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
"""
class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)