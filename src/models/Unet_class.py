"""
U-Net with a binary classification head for watermarking.
"""
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch


class UNetClass(nn.Module):
    def __init__(self, im_size=128, classes=2, backbone="mobilenet_v2", c_in=3):
        """
        Unet mode with a classification head.   

        :param im_size: size of the input image
        :type im_size: int
        :param classes: number of classification classes
        :type classes: int
        :param backbone: the type of backbone for the U-net model
        :type backbone: str
        :param c_in: number of color channels on the input image
        :type c_in: int
        """
        super(UNetClass, self).__init__()
        model = smp.Unet(
            encoder_name=backbone,       # backbone
            encoder_weights="imagenet",  # pre-trained weights
            in_channels=c_in,            # model input channels
            classes=classes
        )
        #
        # cut off segmentation head
        #
        self.encoder = model.encoder
        self.decoder = model.decoder
        #
        # Add reconstruction layer
        #
        self.reconstruction_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        #
        # Add watermarking layer
        #
        self.watermarking_head = nn.Linear(im_size*im_size*16, 2)
        #
        # Activation function
        #
        self.activation_recon = nn.Tanh()


    def forward(self, x):      
        x = self.encoder(x)
        x = self.decoder(*x) # have to star here
        x_hat = self.reconstruction_head(x)
        x_hat = self.activation_recon(x_hat)
        #
        # flatten and then class predict
        #
        x = x.flatten(1)
        y_pred = self.watermarking_head(x)

        return x_hat, y_pred
    

    def generate(self, x):    
        out, y_pred = self.forward(x)
        #
        # current tanh activate is [-1, 1].
        # Add 1 --> [0, 2], then divide by 2 gets you to [0, 1]
        #
        out = (out+1)/2   
        
        return torch.clamp(out, 0, 1), torch.argmax(y_pred, dim=1)
