"""


Notes:
- Supported models and their backbones
    > unet: resnet34, resnet50, mobilenet_v2
    > linknet: resnet34, resnet50, mobilenet_v2
    > fpn: resnet34, resnet50, mobilenet_v2
    > pspnet: resnet34, resnet50, mobilenet_v2
    > deeplabv3: resnet, mobilenet
    > lraspp: mobilenet
    > fcn: resnet50, resnet101
"""
from datasets.data_cifar_triggers import CifarTriggerDataset
from datasets.data_imagenet_triggers import ImageNetTriggerDataset
from datasets.data_cifar import CifarDataset
from datasets.data_imagenet import ImageNetDataset
from datasets.data_coco import COCODataset
from datasets.data_clwd import CLWDDataset
from datasets.data_clwd_triggers import CLWDTriggerDataset
from datasets.data_coco_triggers import COCOTriggerDataset
from datasets.data_coco_semseg_triggers import COCOSemsegTriggerDataset
from datasets.data_coco_semseg import COCOSemsegDataset
from datasets.data_cifar_class import CifarClassDataset
from datasets.data_optic_disc_semseg_triggers import OpticDiscSemsegTriggerDataset
from datasets.data_optic_disc_semseg_triggers_v2 import OpticDiscSemsegTriggerDatasetV2
from datasets.data_optic_disc_semseg import OpticDiscSemsegDataset
from datetime import datetime
from torch import utils
import torch.nn.utils.prune as prune
import torch
import glob
import os
import torchvision.models.segmentation as tch_seg

# model imports
from models import Unet, Linknet, Deeplab, Lraspp, FCN, FPN, PSPNet, PAN, \
    Unet_semseg, Linknet_semseg, FPN_semseg, PSPNet_semseg, PAN_semseg, \
    Deeplab_semseg, Lraspp_semseg

# find valid backbone names here: 
# https://github.com/chsasank/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/encoders
# update as needed.
VALID_SMP_BACKBONES = ['resnet34', 'resnet50', 'mobilenet_v2']

def get_denoise_model(model_type: str, backbone_type: str) -> torch.nn.Module:
    """Get the denoising model to use for training."""
    # make sure backbone exists for specified model
    if model_type in ["unet", "linknet"]:
        assert backbone_type in VALID_SMP_BACKBONES, "Please specify a valid backbone."
    
    # load the model
    if model_type == "unet":
        model = Unet.UNetModel(
            backbone=backbone_type, 
            c_in=3
        )
    elif model_type == "linknet":
        model = Linknet.LinknetModel(
            backbone=backbone_type,
            c_in=3
        )
    elif model_type == "fpn":
        model = FPN.FPNModel(
            backbone=backbone_type,
            c_in=3
        )
    elif model_type == "pspnet":
        model = PSPNet.PSPNetModel(
            backbone=backbone_type,
            c_in=3
        )
    elif model_type == "pan":
        model = PAN.PANModel(
            backbone=backbone_type,
            c_in=3
        )
    elif model_type == "deeplabv3": # used this i think
        if backbone_type == "resnet":
            backbone = tch_seg.deeplabv3_resnet50(weights=tch_seg.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
            head_dim = 2048
        elif backbone_type == "mobilenet":
            backbone = tch_seg.deeplabv3_mobilenet_v3_large(weights=tch_seg.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
            head_dim = 960
        else:
            raise ValueError("Valid backbones for Deeplab are resnet and mobilenet.")
        model = Deeplab.DeeplabModel(
            backbone=backbone,
            head_dim=head_dim
        )
    elif model_type == "lraspp":
        if backbone_type == "mobilenet":
            backbone = tch_seg.lraspp_mobilenet_v3_large(weights=tch_seg.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
        else:
            raise ValueError("Valid backbones for LRASPP are mobilenet.")
        model = Lraspp.LRASPPModel(
            backbone=backbone
        )
    elif model_type == "fcn":
        if backbone_type == "resnet50":
            backbone = tch_seg.fcn_resnet50(weights=tch_seg.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        elif backbone_type == "resnet101":
            backbone = tch_seg.fcn_resnet101(weights=tch_seg.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        else:
            raise ValueError("Valid backbones for FCN are resnet50 and resnet101.")
        model = FCN.FCNModel(
            backbone=backbone
        )
    else:
        raise ValueError("Invalid model type.")

    return model


def get_semseg_model(model_type: str, backbone_type: str, num_classes: int) -> torch.nn.Module:
    """Get the semantic segmentation model to use for training."""
    # make sure backbone exists for specified model
    if model_type in ["unet", "linknet"]:
        assert backbone_type in VALID_SMP_BACKBONES, "Please specify a valid backbone."
    
    # load the model
    if model_type == "unet":
        model = Unet_semseg.UNetSemseg(
            classes=num_classes,
            backbone=backbone_type, 
            c_in=3
        )
    elif model_type == "linknet":
        model = Linknet_semseg.LinknetSemseg(
            classes=num_classes,
            backbone=backbone_type,
            c_in=3
        )
    elif model_type == "fpn":
        model = FPN_semseg.FPNSemseg(
            classes=num_classes,
            backbone=backbone_type,
            c_in=3
        )
    elif model_type == "pspnet":
        model = PSPNet_semseg.PSPNetSemseg(
            classes=num_classes,
            backbone=backbone_type,
            c_in=3
        )
    elif model_type == "pan":
        model = PAN_semseg.PANSemseg(
            classes=num_classes,
            backbone=backbone_type,
            c_in=3
        )
    elif model_type == "deeplabv3": # used this i think
        if backbone_type == "resnet":
            backbone = tch_seg.deeplabv3_resnet50(weights=tch_seg.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
            head_dim = 2048
            scale=8.0
        elif backbone_type == "mobilenet":
            backbone = tch_seg.deeplabv3_mobilenet_v3_large(weights=tch_seg.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
            head_dim = 960
            scale=16.0
        else:
            raise ValueError("Valid backbones for Deeplab are resnet and mobilenet.")
        model = Deeplab_semseg.DeeplabSemseg(
            backbone=backbone,
            head_dim=head_dim,
            scale=scale,
            classes=num_classes
        )
    elif model_type == "lraspp":
        if backbone_type == "mobilenet":
            backbone = tch_seg.lraspp_mobilenet_v3_large(weights=tch_seg.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
        else:
            raise ValueError("Valid backbones for LRASPP are mobilenet.")
        model = Lraspp_semseg.LRASPPSemseg(
            backbone=backbone,
            classes=num_classes
        )
    else:
        raise ValueError("Invalid model type.")

    return model


def load_checkpoint(model, checkpoint):
    """Load a checkpoint for the model.

    :param model: the model to load the checkpoint into
    :type model: U-Net, U-NetSemseg, U-NetClass
    :param checkpoint: the checkpoint file to load
    :type checkpoint: str
    """
    print("Loading pretrained checkpoint ... ")
    checkpoints = sorted(glob.glob(os.path.join(checkpoint, "**.pth")))
    checkpoint = torch.load(checkpoints[-1]) # take the most recent
    try:
        model_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(model_state_dict)
    except:
        model = checkpoint

    return model

def prune_model_global_unstructured(model, pruning_ratio=0.1):
    """Prune a model.

    :param model: the model to prune
    :type model: U-Net, U-NetSemseg, U-NetClass
    :param pruning_ratio: The amount of the model to prune
    :type pruning_ratio: float
    """
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio,
    )
    
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')


def load_data(mode, args):
    """Load datasets.

    :param mode: training or test/validation
    :type mode: str
    :param args: data loader module arguments
    :type args: Argparse
    :return: Dataloader of images
    :rtype: torch.utils.data.DataLoader
    """
    #
    # Dataset
    #
    print("Curating dataset ...")
    if args.no_trigger:
        print("Using normal dataset for training ... ")
        if args.dataset == "cifar":
            dataset = CifarDataset(
                mode=mode,
                num_images=args.num_images,
                imsize=args.imsize
            )
        elif args.dataset == "imagenet":
            dataset = ImageNetDataset(
                path=args.data_path,
                mode=mode,
                num_images=args.num_images,
                imsize=args.imsize
            )
        elif args.dataset == "clwd":
            dataset = CLWDDataset(
                path=args.data_path,
                mode=mode,
                num_images=args.num_images,
                imsize=args.imsize
            )
        elif args.dataset == "coco":
            dataset = COCODataset(
                path=args.data_path,
                mode=mode,
                num_images=args.num_images,
                imsize=args.imsize
            )

    else:
        print("Using trigger dataset for training ... ")
        if args.dataset == "cifar":
            dataset = CifarTriggerDataset(
                mode=mode,
                num_images=args.num_images,
                trigger_c=args.trigger_c,
                trigger_pos=args.trigger_pos,
                trigger_s=args.trigger_s,
                response_c=args.response_c,
                response_pos=args.response_pos,
                response_s=args.response_s,
                image_signature=args.image_signature,
                imsize=args.imsize
            )
        elif args.dataset == "imagenet":
            dataset = ImageNetTriggerDataset(
                path=args.data_path,
                mode=mode,
                num_images=args.num_images,
                trigger_c=args.trigger_c,
                trigger_pos=args.trigger_pos,
                trigger_s=args.trigger_s,
                response_c=args.response_c,
                response_pos=args.response_pos,
                response_s=args.response_s,
                image_signature=args.image_signature,
                noise_trigger=args.noise_trigger,
                steg_trigger=args.steg_trigger,
                imsize=args.imsize
            )
        elif args.dataset == "clwd":
            dataset = CLWDTriggerDataset(
                path=args.data_path,
                mode=mode,
                num_images=args.num_images,
                trigger_c=args.trigger_c,
                trigger_pos=args.trigger_pos,
                trigger_s=args.trigger_s,
                response_c=args.response_c,
                response_pos=args.response_pos,
                response_s=args.response_s,
                image_signature=args.image_signature,
                imsize=args.imsize
            )
        elif args.dataset == "coco":
            dataset = COCOTriggerDataset(
                path=args.data_path,
                mode=mode,
                num_images=args.num_images,
                trigger_c=args.trigger_c,
                trigger_pos=args.trigger_pos,
                trigger_s=args.trigger_s,
                response_c=args.response_c,
                response_pos=args.response_pos,
                response_s=args.response_s,
                noise_trigger=args.noise_trigger,
                image_signature=args.image_signature,
                steg_trigger=args.steg_trigger,
                imsize=args.imsize
            )
                
    data_loader = utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=2, shuffle=True if mode == "train" else False, pin_memory=True, persistent_workers=True)

    return data_loader


def load_semseg_data(mode, args):
    """Load datasets.

    :param mode: training or test/validation
    :type mode: str
    :param args: data loader module arguments
    :type args: Argparse
    :return: Dataloader of images
    :rtype: torch.utils.data.DataLoader
    """
    #
    # Dataset
    #
    print("Curating dataset ...")
    if args.no_trigger:
        print("Using normal dataset for training ... ")
        if args.dataset == "coco":
            dataset = COCOSemsegDataset(
                path=args.data_path,
                mode=mode,
                num_images=args.num_images,
            )
        elif args.dataset in ["optic"]:
            dataset = OpticDiscSemsegDataset(
                path=args.data_path,
                mode=mode,
                num_images=args.num_images,
                imsize=args.imsize,
            )
    else:
        print("Using trigger dataset for training ... ")
        if args.dataset == "coco":
            dataset = COCOSemsegTriggerDataset(
                path=args.data_path,
                mode=mode,
                num_images=None,
                trigger_c=args.trigger_c,
                trigger_pos=args.trigger_pos,
                trigger_s=args.trigger_s,
                response_c=args.response_c,
                response_pos=args.response_pos,
                response_s=args.response_s,
                noise_trigger=args.noise_trigger,
                image_signature=args.image_signature,
                steg_trigger=args.steg_trigger,
            )
        elif args.dataset == "optic":
            dataset = OpticDiscSemsegTriggerDatasetV2(
                path=args.data_path,
                mode=mode,
                num_images=None,
                trigger_c=args.trigger_c,
                trigger_pos=args.trigger_pos,
                trigger_s=args.trigger_s,
                imsize=args.imsize,
                noise_trigger=args.noise_trigger,
                steg_trigger=args.steg_trigger,
            )
                
    data_loader = utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=2, shuffle=True if mode == "train" else False, pin_memory=True, persistent_workers=True)

    return data_loader


def load_class_data(mode, args):
    """Load datasets.

    :param mode: training or test/validation
    :type mode: str
    :param args: data loader module arguments
    :type args: Argparse
    :return: Dataloader of images
    :rtype: torch.utils.data.DataLoader
    """
    #
    # Dataset
    #
    print("Curating dataset ...")
    if args.no_trigger:
        print("Using normal dataset for training ... ")
        if args.dataset == "cifar":
            dataset = CifarDataset(
                mode=mode,
                num_images=None
            )
    else:
        print("Using classification dataset for training ... ")
        dataset = CifarClassDataset(
                mode=mode,
                num_images=None,
                trigger_c=args.trigger_c,
                trigger_pos=args.trigger_pos,
                trigger_s=args.trigger_s
        )
                
    data_loader = utils.data.DataLoader(dataset, batch_size=32, num_workers=2, shuffle=True if mode == "train" else False, pin_memory=True, persistent_workers=True)

    return data_loader


def get_name(args):
    """Generate an experiment name from the file arguments.

    :param args: arguments used during training/test
    :type args: Argparse
    :return: the name of the experiment file
    :rtype: str
    """
    now = datetime.now()
    date_time_str = now.strftime("%m%d-%H%M")
    pos = args.trigger_pos.split("_")
    if len(pos) == 1:
        trigger_name = f"{args.trigger_c[:2]}-{args.trigger_pos[0]}{args.trigger_pos[1]}-{args.trigger_s}"
    else:
        trigger_name = f"{args.trigger_c[:2]}-{pos[0][0]}{pos[1][0]}-{args.trigger_s}"
    pos = args.response_pos.split("_")
    if len(pos) == 1:
        signature_name = f"{args.response_c[:2]}-{args.response_pos[0]}{args.response_pos[1]}-{args.response_s}"
    else:
        signature_name = f"{args.response_c[:2]}-{pos[0][0]}{pos[1][0]}-{args.response_s}"
    mode = "ftune" if args.no_trigger else "train"

    model_name = f"{args.model_type}_{date_time_str}_{trigger_name}_{signature_name}_{mode}.pth"
    expr_name = f"{args.model_type}_{trigger_name}_{signature_name}"

    return model_name, expr_name


def get_name_class(args):
    """Gnerate an experiment name from the file arguments.

    :param args: arguments used during training/test
    :type args: Argparse
    :return: the name of the experiment file
    :rtype: str
    """
    now = datetime.now()
    date_time_str = now.strftime("%m%d-%H%M")
    pos = args.trigger_pos.split("_")
    if len(pos) == 1:
        trigger_name = f"{args.trigger_c[:2]}-{args.trigger_pos[0]}{args.trigger_pos[1]}-{args.trigger_s}"
    else:
        trigger_name = f"{args.trigger_c[:2]}-{pos[0][0]}{pos[1][0]}-{args.trigger_s}"
    
    model_name = f"unet_{date_time_str}_classbase_{trigger_name}.pth"
    expr_name = f"unet_classbase_{trigger_name}"

    return model_name, expr_name


def pretty_print_args(args):
    """Print arguments.

    :param args: the arguments used during trainig/test to print
    :type args: Argparse
    """
    print("------------------------------")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("------------------------------\n\n")
    
