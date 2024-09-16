"""
LRASPP model implementation for semantic segmentation.
"""

import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch import Tensor


class LRASPPSemseg(nn.Module):
    def __init__(self, backbone, classes):
        super(LRASPPSemseg, self).__init__()
        #
        # Cut off classifier head from backbone model
        #
        self.backbone = backbone.backbone
        #
        # Add segmentation head
        #
        self.segmentation_head = LRASPPHead(
            low_channels=40, 
            high_channels=960, 
            output_channels=classes, 
            inter_channels=256
        )

        
    def forward(self, input):
        # extract features from backbone
        x = self.backbone(input)
        # semseg
        semseg = self.segmentation_head(x)
        semseg = F.interpolate(semseg, size=input.shape[-2:], mode="bilinear", align_corners=False)

        return semseg

"""
From https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py
"""
class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, output_channels: int, inter_channels: int) -> None:
        super().__init__()
        # Use Conv-BatchNorm-ReLU blocks for feature transformation
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        # Sigmoid scale layer remains to modulate the features
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # Reconstruction heads for low and high level features
        self.low_reconstruct = nn.Conv2d(low_channels, output_channels, 1)
        self.high_reconstruct = nn.Conv2d(inter_channels, output_channels, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        # Process high-level features
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        # Combine low and high level features, passed through reconstruct layers
        out = self.low_reconstruct(low) + self.high_reconstruct(x)
        return out