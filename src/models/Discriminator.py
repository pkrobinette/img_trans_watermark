"""
From: Model Watermarking for Image Processing Networks
Code: https://github.com/ZJZAC/Deep-Model-Watermarking

"""

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

        
class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """
        Construct a 1x1 PatchGAN discriminator
        
        :param input_nc: the number of channels in input images
        :type input_nc: int
        :param ndf: the number of filters in the last conv layer
        :type ndf: int
        :param norm_layer: normalization layer
        :param norm_layer: nn.BatchNorm2d
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):    
        return self.net(input)

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Construct a PatchGAN discriminator

        :param input_nc: the number of channels in input images
        :type input_nc: int
        :param ndf: the number of filters in the last conv layer
        :type ndf: int 
        :param n_layers: the number of conv layers in the discriminator
        :type n_layers: int
        :param norm_layer: normalization layer
        :param norm_layer: nn.BatchNorm2d
        """
        super(PatchDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, requires_grad=True):
        """Placeholder

        :param in_channels: Number of input channels
        :type in_channels: int
        :param requires_grad: If true, weights are updated
        :type requires_grad: bool
        """
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block.

            :param in_filters: number of input filters
            :type in_filters: int
            :param out_filters: number of output filters
            :type out_filters: int
            :param stride: convolutional stride
            :type stride: int
            :param normalize: whether to normalize the layers
            :type normalize: bool
            :return: layers
            :rtype: list[layers]
            """            
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:   
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers.extend(discriminator_block(in_channels, 64, 2, False))
        layers.extend(discriminator_block(64, 128, 2, True))
        layers.extend(discriminator_block(128, 256, 2, True))
        layers.extend(discriminator_block(256, 512 , 2, True))
        layers.append(nn.Conv2d(512, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)



        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False   


    def forward(self, img):
        return self.model(img)
