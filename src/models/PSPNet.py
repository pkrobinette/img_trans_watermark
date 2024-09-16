"""
PSPNet model implementation for denoising.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn
import torch


class PSPNetModel(nn.Module):
    def __init__(self, backbone="mobilenet_v2", c_in=3):
        """A PSPNet implementation.

        :param backbone: the type of backbone for the PSPNet model
        :type backbone: str
        :param c_in: number of color channels on the input image
        :type c_in: int
        """    
        super(PSPNetModel, self).__init__()
        model = smp.PSPNet(
            encoder_name=backbone,       # backbone
            encoder_weights="imagenet",  # pre-trained weights
            in_channels=c_in,            # model input channels
        )
        #
        # cut off segmentation head
        #
        self.encoder = model.encoder
        self.decoder = model.decoder
        #
        # Add denoising layer
        #
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=8.0, mode='bilinear', align_corners=False)  # align_corners is often set to False for bilinear upsampling
        )
        #
        # Activation function
        #
        self.activation = nn.Tanh()


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(*x) # have to star here
        x = self.output_conv(x)
        x = self.activation(x)

        return x
    

    def generate(self, x):
        out = self.forward(x)
        #
        # current tanh activate is [-1, 1].
        # Add 1 --> [0, 2], then divide by 2 gets you to [0, 1]
        #
        out = (out+1)/2   
        
        return torch.clamp(out, 0, 1)