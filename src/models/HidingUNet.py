"""
From: Model Watermarking for Image Processing Networks
Code: https://github.com/ZJZAC/Deep-Model-Watermarking

"""

import functools
import torch
import torch.nn as nn
from torch.autograd import Variable

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck

def gaussian_noise(tensor, mean=0, stddev=0.1): 
    """
    Create gaussian noise of size tensor.size().

    :param tensor: tensor size to create noise from
    :type tensor: torch.Tensor
    :param mean: mean of the gaussian noise
    :type mean: float
    :param stddev: standard deviation of the guassian noise
    :type stddev: float
    """ 
    noise = torch.nn.init.normal(torch.Tensor(tensor.size()), 0, 0.1)
    return Variable(tensor + noise)

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid,  requires_grad=True):
        """
        Unet Generator.

        :param input_nc: the number of channels in input images
        :type input_nc: int
        :param output_nc: the number of output channels
        :type output_nc: int
        :param num_downs: number of downsampling/upsampling steps. This defines the depth of the U-Net architecture.
        :type num_downs: int
        :param ngf: number of filters in the outermost layer of the generator. Default is 64.
        :type ngf: int
        :param norm_layer: normalization layer used in the network (e.g., BatchNorm2d).
        :type norm_layer: torch.nn.Module
        :param use_dropout: if True, dropout layers are added to the model. Default is False.
        :type use_dropout: bool
        :param output_function: the final layer's activation function (e.g., nn.Sigmoid).
        :type output_function: torch.nn.Module
        :param requires_grad: if false, the model's parameters will not be updated during training.
        :type requires_grad: bool
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False   


    def forward(self, input):

        # x = self.model(input)
        # x_n = gaussian_noise(x.data, 0, 0.1)

        # return x_n
        return self.model(input)

class UnetGenerator_IN(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                norm_layer=nn.InstanceNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        """
        Unet Generator.

        :param input_nc: the number of channels in input images
        :type input_nc: int
        :param output_nc: the number of output channels
        :type output_nc: int
        :param num_downs: number of downsampling/upsampling steps. This defines the depth of the U-Net architecture.
        :type num_downs: int
        :param ngf: number of filters in the outermost layer of the generator. Default is 64.
        :type ngf: int
        :param norm_layer: normalization layer used in the network (e.g., BatchNorm2d).
        :type norm_layer: torch.nn.Module
        :param use_dropout: if True, dropout layers are added to the model. Default is False.
        :type use_dropout: bool
        :param output_function: the final layer's activation function (e.g., nn.Sigmoid).
        :type output_function: torch.nn.Module
        """
        super(UnetGenerator_IN, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock_IN(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock_IN(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock_IN(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_IN(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_IN(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_IN(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block

    def forward(self, input):


        return self.model(input)




# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module): 
    def __init__(self, outer_nc, inner_nc, input_nc=None,submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        """
        Initializes a U-Net skip connection block.

        :param outer_nc: Number of channels in the output images for this block.
        :type outer_nc: int
        :param inner_nc: Number of channels in the input images for this block.
        :type inner_nc: int
        :param input_nc: Number of channels in the input images. If None, defaults to outer_nc.
        :type input_nc: int, optional
        :param submodule: The submodule used for constructing deeper U-Net structures. This is typically another UnetSkipConnectionBlock.
        :type submodule: nn.Module, optional
        :param outermost: If True, this block is treated as the outermost block, affecting its construction, particularly the absence of normalization after the upconvolution.
        :type outermost: bool
        :param innermost: If True, this block is treated as the innermost block, affecting its construction, particularly skipping the downnormalization step.
        :type innermost: bool
        :param norm_layer: The normalization layer to use within the block. Defaults to BatchNorm2d.
        :type norm_layer: torch.nn.Module
        :param use_dropout: If True, dropout is added after the upconvolution step for additional regularization.
        :type use_dropout: bool
        :param output_function: The activation function used at the output of the outermost block. Defaults to Sigmoid.
        :type output_function: torch.nn.Module
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # if norm_layer == 'nn.BatchNorm2d':
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        #     norm_layer =
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d     

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                            stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetSkipConnectionBlock_IN(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        """
        Initializes a U-Net skip connection block with instance normalization.

        :param outer_nc: Number of channels in the output images for this block.
        :type outer_nc: int
        :param inner_nc: Number of channels in the input images for this block.
        :type inner_nc: int
        :param input_nc: Number of channels in the input images. If None, defaults to outer_nc. This parameter allows for flexibility in connecting blocks with different numbers of channels.
        :type input_nc: int, optional
        :param submodule: Nested U-Net block that forms the submodule of this block. Allows for recursive construction of the U-Net architecture.
        :type submodule: nn.Module, optional
        :param outermost: Specifies if this block is the outermost layer. Influences the construction of the block, especially the use of the output function and lack of normalization on the output.
        :type outermost: bool
        :param innermost: Specifies if this block is the innermost layer. Influences the construction of the block, especially skipping the normalization on the downsampling path.
        :type innermost: bool
        :param norm_layer: Normalization layer to use within the block. Defaults to instance normalization (nn.InstanceNorm2d).
        :type norm_layer: torch.nn.Module
        :param use_dropout: Enables dropout in the block for regularization, typically used in the non-outermost blocks.
        :type use_dropout: bool
        :param output_function: Activation function used at the output of the outermost block. Defaults to Sigmoid, but can be set to any appropriate torch.nn function.
        :type output_function: torch.nn.Module
        """
        super(UnetSkipConnectionBlock_IN, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # if norm_layer == 'nn.BatchNorm2d':
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        #     norm_layer =
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d     

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
