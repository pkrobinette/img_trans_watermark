"""
POST-Spatial Watermarking: The watermark model is used after the
target model.

From: Model Watermarking for Image Processing Networks
Code: https://github.com/ZJZAC/Deep-Model-Watermarking

"""

import argparse
import os
import shutil
import socket
import time
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision
import transformed as transforms
from models.HidingUNet import UnetGenerator
from models.Discriminator import Discriminator
from models.HidingRes import HidingRes
import numpy as np
from PIL import Image
import lightning.pytorch as pl
import sys
from utils import load_data, get_name, pretty_print_args, \
    prune_model_global_unstructured, get_denoise_model
from modules.module_denoise_watermark import DenoiseTriggerModule
from metrics.metrics_reconstruct import calculate_image_metrics
from collections import defaultdict
import pickle
from torchvision.utils import save_image

## Color map for adding boxes
COLORS = {
    "purple": (128, 0, 128),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "pink": (255, 192, 203),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0)
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar", help='dataset used for training')
parser.add_argument("--data_path", type=str, default=None, help='path of dataset')
parser.add_argument("--num_images", type=int, default=None, help='Number of images to use')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--imsize', type=int, default=128, help='the number of frames')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--epochs_target_model', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10, help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--gpu', action='store_true', help='enables cuda')
parser.add_argument('--num_devices', type=int, default=None, help='Number of devices to use for training')
parser.add_argument('--Hnet', default='', help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='', help="path to Revealnet (to continue training)")
parser.add_argument('--Dnet', default='', help="path to Discriminator (to continue training)")
parser.add_argument("--checkpoint", type=str, default=None, help="training checkpoint to finetune")
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')
parser.add_argument("--num_test_save", type=int, default=10, help="number of test images to save; will be 2x (normal and triggers)")
parser.add_argument('--datasets', type=str, default='cifar10', help='name of dataset')
parser.add_argument("--trigger_c", type=str, default="purple", help="color of trigger")
parser.add_argument("--trigger_s", type=str, default="small", help="size of the trigger")
parser.add_argument("--response_c", type=str, default="green", help="color or signature")
parser.add_argument("--response_s", type=str, default="small", help="size of the signature")
parser.add_argument("--trigger_pos", type=str, default="top_right", help="position of trigger")
parser.add_argument("--response_pos", type=str, default="top_right", help="position of signature")

#read secret image
parser.add_argument('--image_signature', type=str, default=None, help='secret folder')
parser.add_argument('--clean', type=str, default="src/datasets/signatures/clean.png")

#hyperparameter of loss

parser.add_argument('--beta', type=float, default=1,
                    help='hyper parameter of beta :secret_reveal err')
parser.add_argument('--betagan', type=float, default=1,
                    help='hyper parameter of beta :gans weight')
parser.add_argument('--betagans', type=float, default=0.01,
                    help='hyper parameter of beta :gans weight')
parser.add_argument('--betapix', type=float, default=0,
                    help='hyper parameter of beta :pixel_loss weight')

parser.add_argument('--betamse', type=float, default=10000,
                    help='hyper parameter of beta: mse_loss')
parser.add_argument('--betacons', type=float, default=1,
                    help='hyper parameter of beta: consist_loss')
parser.add_argument('--betaclean', type=float, default=1,
                    help='hyper parameter of beta: clean_loss')
parser.add_argument('--betacleanA', type=float, default=1,
                    help='hyper parameter of beta: clean_loss')
parser.add_argument('--betacleanB', type=float, default=1,
                    help='hyper parameter of beta: clean_loss')
parser.add_argument('--betavgg', type=float, default=0,
                    help='hyper parameter of beta: vgg_loss')
parser.add_argument('--num_downs', type=int, default= 7 , help='nums of  Unet downsample')
parser.add_argument('--clip', action='store_true', help='clip container_img')
parser.add_argument("--attack", type=str, default=None, help="attack type")
# !! this is set as default true, because the 
# triggers are made in this file.
parser.add_argument("--trigger", action='store_true', help="use non-trigger dataset")
parser.add_argument('--save_path', type=str, help='name to save file')
parser.add_argument('--metrics_fname', type=str, default='test_results', help='the name of the test results file')





def main():
    ############### define global parameters ###############
    global args, optimizerH, optimizerR, optimizerD, writer, logPath, schedulerH, schedulerR
    global val_loader, smallestLoss,  mse_loss, gan_loss, pixel_loss, patch, criterion_GAN, criterion_pixelwise,vgg, vgg_loss

    args = parser.parse_args()
    if torch.cuda.is_available() and not args.gpu:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --gpu")

    cudnn.benchmark = True

    ############  create the dirs to save the result #############

    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    experiment_dir = args.save_path
    args.outckpts = experiment_dir + "/checkPoints"
    args.trainpics = experiment_dir + "/trainPics"
    args.validationpics = experiment_dir + "/validationPics"
    args.outlogs = experiment_dir + "/trainingLogs"
    args.outcodes = experiment_dir + "/codes"
    args.testPics = experiment_dir + "/testPics"
    args.runfolder = experiment_dir + "/run"

    if not os.path.exists(args.outckpts):
        os.makedirs(args.outckpts)
    if not os.path.exists(args.trainpics):
        os.makedirs(args.trainpics)
    if not os.path.exists(args.validationpics):
        os.makedirs(args.validationpics)
    if not os.path.exists(args.outlogs):
        os.makedirs(args.outlogs)
    if not os.path.exists(args.outcodes):
        os.makedirs(args.outcodes)
    if not os.path.exists(args.runfolder):
        os.makedirs(args.runfolder)              
    if (not os.path.exists(args.testPics)) and args.test != '':
        os.makedirs(args.testPics)

    logPath = args.outlogs + '/%s_%d_log.txt' % (args.dataset, args.batch_size)

    print_log(str(args), logPath)
    writer = SummaryWriter(log_dir=args.runfolder, comment='**' + args.save_path)
    # update for data loader 
    # pre and post methods do not use triggers
    if args.trigger == False:
        args.no_trigger = True
    #
    # assert clean and secret path exist
    #
    if args.image_signature is not None:
        assert os.path.exists(os.path.join("src/datasets/signatures", f"{args.image_signature}.jpg")), "args.image_signature does not exist!"
    assert os.path.exists(args.clean), "args.clean path does not exist!"   	


    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs= args.num_downs, output_function=nn.Sigmoid)
    Hnet.cuda()
    Hnet.apply(weights_init)

    Rnet = HidingRes(in_c=3, out_c=3)
    Rnet.cuda()
    Rnet.apply(weights_init)

    Dnet = Discriminator(in_channels=3)
    Dnet.cuda()
    Dnet.apply(weights_init)

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, args.imsize // 2 ** 4, args.imsize // 2 ** 4)

    # setup optimizer
    optimizerH = optim.Adam(Hnet.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

    optimizerR = optim.Adam(Rnet.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)

    optimizerD = optim.Adam(Dnet.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.2, patience=5, verbose=True)


    if args.Hnet != "":
        Hnet.load_state_dict(torch.load(args.Hnet))
    if args.num_devices > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()

    if args.Rnet != '':
        Rnet.load_state_dict(torch.load(args.Rnet))
    if args.num_devices > 1:
        Rnet = torch.nn.DataParallel(Rnet).cuda()

    if args.Dnet != '':
        Dnet.load_state_dict(torch.load(args.Dnet))
    if args.num_devices > 1:
        Dnet = torch.nn.DataParallel(Dnet).cuda()


    # define loss
    mse_loss = nn.MSELoss().cuda()
    criterion_GAN = nn.MSELoss().cuda()
    criterion_pixelwise = nn.L1Loss().cuda()
    # -----------------------
    # Target Model
    # -----------------------
    unet = get_denoise_model(
        model_type="unet",
        backbone_type="mobilenet_v2"
    )
    #
    # Load checkpoint if indicated
    #
    if args.checkpoint is not None:
        print("Loading pretrained checkpoint ... ")
        # checkpoints = sorted(glob.glob(os.path.join(args.checkpoint, "**")))
        checkpoint = torch.load(os.path.join(args.checkpoint, "target_model.pth")) # take the most recent
        try:
            model_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            unet.load_state_dict(model_state_dict)
        except:
            unet = checkpoint

    target_model = DenoiseTriggerModule(
        model=unet,         # model
        alpha=1.0,          # weight of watermarking loss
        lr=0.001,           # learning rate
        num_images_save=10, # number of images to save during testing
        save_path=os.path.join(experiment_dir, "images")
    )
    # -----------------------
    # Model ATTACKS
    # -----------------------
    target_model_train = True if not args.test else False
    watermark_train = True if not args.test else False
    if args.attack is not None:
        if args.attack == "prune10":
            print("\n\nATTACK: Pruning 10% of model ..........................................................\n\n")
            prune_model_global_unstructured(target_model.model, pruning_ratio=0.1)
            print("Model saved to: ", os.path.join(experiment_dir, "target_model_attack_prune10"))
            torch.save(target_model.model, os.path.join(experiment_dir, "target_model_attack_prune10.pth"))  
            target_model_train = False
        elif args.attack == "prune40":
            print("\n\nATTACK: Pruning 40% of model ..........................................................\n\n")
            prune_model_global_unstructured(target_model.model, pruning_ratio=0.4)
            print("Model saved to: ", os.path.join(experiment_dir, "target_model_attack_prune40"))
            torch.save(target_model.model, os.path.join(experiment_dir, "target_model_attack_prune40.pth"))  
            target_model_train = False
        elif args.attack == "ftune1":
            print("\n\nATTACK: Finetuning 1 epoch ..........................................................")
            args.epochs_target_model = 1
            args.no_trigger = True
        elif args.attack == "ftune5":
            print("\n\nATTACK: Finetuning 5 epochs ..........................................................")
            args.epochs_target_model = 5
            args.no_trigger = True
        elif args.attack == "overwrite":
            print("\n\nATTACK: Overwriting ..........................................................")
            args.epochs_target_model = 5
            args.no_trigger = False
            args.image_signature = "flower" 

        watermark_train = False

    train_loader = load_data("train", args)
    args.num_images = 200
    val_loader = load_data("test", args)    # small sample (200 images)
    # ------------------------------
    # Train image processing attack
    # ------------------------------
    if target_model_train:
        print_log("\n\ntraining the target model (reconstruction) .......................................................\n\n", logPath)
        #
        # Set up trainer
        #
        trainer = pl.Trainer(
            max_epochs=args.epochs_target_model,
            devices=args.num_devices,
            accelerator="gpu" if args.gpu else None,
        )
        #
        # Train
        #
        print("Training ...")
        trainer.fit(model=target_model, train_dataloaders=train_loader)
        #
        # Save model
        #
        if not args.attack:
            print("Model saved to: ", os.path.join(experiment_dir, "target_model"))
            torch.save(target_model.model, os.path.join(experiment_dir, "target_model.pth"))
        else:
            print("Model saved to: ", os.path.join(experiment_dir, f"target_model_attack_ftune{args.epochs_target_model}"))
            torch.save(target_model.model, os.path.join(experiment_dir, f"target_model_attack_ftune{args.epochs_target_model}.pth"))
    # ------------------------------------
    # turn grad off for image_processing
    # ------------------------------------
    target_model = target_model.model
    target_model = target_model.to('cuda')
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False
    # ------------------------------------
    # Training Watermark
    # -----------------------------------
    if watermark_train:
        smallestLoss = 10000
        print_log("\n\ntraining for watermarking .......................................................\n\n", logPath)
        for epoch in range(args.epochs):
            ######################## train ##########################################
            train(train_loader, epoch, Hnet=Hnet, Rnet=Rnet, Dnet=Dnet, target_model=target_model)

            ####################### validation  #####################################
            val_hloss, val_rloss, val_r_mseloss, val_r_consistloss, val_dloss, val_fakedloss, val_realdloss, val_Ganlosses, val_Pixellosses,vgg_loss, val_sumloss = validation(val_loader,  epoch, Hnet=Hnet, Rnet=Rnet, Dnet=Dnet, target_model=target_model)

            ####################### adjust learning rate ############################
            schedulerH.step(val_sumloss)
            schedulerR.step(val_rloss)
            schedulerD.step(val_dloss)

            # save the best model parameters
            if val_sumloss < globals()["smallestLoss"]:
                globals()["smallestLoss"] = val_sumloss

                torch.save(Hnet.state_dict(),
                               '%s/netH_epoch_%d,sumloss=%.6f,Hloss=%.6f.pth' % (
                                   args.outckpts, epoch, val_sumloss, val_hloss))
                torch.save(Rnet.state_dict(),
                               '%s/netR_epoch_%d,sumloss=%.6f,Rloss=%.6f.pth' % (
                                   args.outckpts, epoch, val_sumloss, val_rloss))
                torch.save(Dnet.state_dict(),
                               '%s/netD_epoch_%d,sumloss=%.6f,Dloss=%.6f.pth' % (
                                   args.outckpts, epoch, val_sumloss, val_dloss))
        # ----------------------------------------
        # Save Model
        # ---------------------------------------                        
        torch.save(Hnet.state_dict(), os.path.join(experiment_dir, 'Hnet.pth'))
        torch.save(Rnet.state_dict(), os.path.join(experiment_dir, 'Rnet.pth'))
        torch.save(Dnet.state_dict(), os.path.join(experiment_dir, 'Dnet.pth'))
        writer.close() 
    # ----------------------------------------
    # Testing
    # ----------------------------------------
    print_log("\n\ntesting .......................................................\n\n", logPath)
    if args.attack == "overwrite":
        args.image_signature = None
        args.response_c = "green"
        args.no_trigger = True
        
    # load the test dataset
    args.num_images = None
    test_loader = load_data("test", args)   # full test dataset

    avg_metrics = test(test_loader, Hnet, Rnet, Dnet, target_model, os.path.join(experiment_dir, "test_images"))
    with open(os.path.join(experiment_dir, f'{args.metrics_fname}.pkl'), 'wb') as file:
        pickle.dump(avg_metrics, file)
    print("-------- Testing Results ---------")
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")
    print("\n\n")


def train(train_loader,  epoch, Hnet, Rnet, Dnet, target_model):
    """Train watermarking module

    :param train_loader: the train loader
    :type train_loader: torch.utils.data.DataLoader
    :param epoch: number of epochs
    :type epoch: int
    :param Hnet: the hiding model; hide a watermark in an image
    :type Hnet: models.HidingUNet
    :param Rnet: the reveal model; reveal a watermark embedded in an image
    :type Rnet: models.HidingRes
    :param Dnet: discriminator model; distinguish between containers (embedded with watermark) and covers (no watermark)
    :type Dnet: models.Discriminator
    :param target_model: The target task model: U-Net reconstruction
    :type target_model: models.Unet
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()  
    Rlosses = AverageMeter()  
    R_mselosses = AverageMeter()
    R_consistlosses = AverageMeter()
    Dlosses = AverageMeter()
    FakeDlosses = AverageMeter()
    RealDlosses = AverageMeter()
    Ganlosses = AverageMeter()
    Pixellosses = AverageMeter()
    Vgglosses =AverageMeter()    
    SumLosses = AverageMeter()  

    # switch to train mode
    Hnet.train()
    Rnet.train()
    Dnet.train()

    # Tensor type
    Tensor = torch.cuda.FloatTensor 

    loader = transforms.Compose([transforms.ToTensor(),
        trans.Resize((args.imsize, args.imsize), antialias=None)])
    clean_img = Image.open(args.clean)
    clean_img = loader(clean_img)  
    if args.image_signature is None:
        secret_img = Image.new('RGB', (args.imsize, args.imsize), COLORS[args.response_c])
    else:
        secret_img = Image.open(f"src/datasets/signatures/{args.image_signature}.jpg")
    secret_img = loader(secret_img)  

    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
        # ---------------------------------------------
        # 1. Run images through the model
        # 2. separate into clean and watermarked
        # 3. Train Hide + GAN
        # 4. Train Reveal 
        # ---------------------------------------------
        data_time.update(time.time() - start_time)
        
        Hnet.zero_grad()
        Rnet.zero_grad()

        if args.gpu:
            data = data.cuda()

        cover_img = target_model(data)
        vgg_loss = mse_loss(cover_img, data)
        
        all_batch_size = int(cover_img.size()[0]) 
        this_batch_size = all_batch_size//2
        cover_img_A = cover_img[:this_batch_size,:,:,:] # these are clean images that are not watermarked
        cover_img_B = cover_img[this_batch_size:,:,:,:] # these are watermarked images

        secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)
        secret_img = secret_img[0:this_batch_size, :, :, :]  
   
        clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)         
        clean_img = clean_img[0:this_batch_size, :, :, :] 

        if args.gpu:
            cover_img = cover_img.cuda()        # [16, 3, 32, 32]
            cover_img_A = cover_img_A.cuda()    # [8, 3, 32, 32]
            cover_img_B = cover_img_B.cuda()    # [8, 3, 32, 32]    
            secret_img = secret_img.cuda()      # [8, 3, 32, 32]
            clean_img = clean_img.cuda()        # [8, 3, 32, 32] 

        # input to watermark
        concat_img = torch.cat([cover_img_B, secret_img], dim=1) # [8, 6, 32, 32 ]
        concat_imgv = Variable(concat_img) 
        cover_imgv = Variable(cover_img_B) # [8, 3, 32, 32 ]
        container_img = Hnet(concat_imgv)  # [8, 6, 32, 32] --> [8, 3, 32, 32]
        A_imgv = Variable(cover_img_A)     # [8, 3, 32, 32]

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((cover_imgv.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((cover_imgv.size(0), *patch))), requires_grad=False)

        pred_fake = Dnet(container_img)
        gan_loss = criterion_GAN(pred_fake, valid)
        pixel_loss = criterion_pixelwise(container_img, cover_imgv)  #l1

        container_img_rgb = container_img.repeat(1,1,1,1)
        cover_imgv_rgb = cover_imgv.repeat(1,1,1,1)
        cover_imgv_rgb.detach()
        #
        # Hide Error
        # MSE(container, cover) + GAN + PixelLoss
        #
        errH = args.betamse * mse_loss(container_img, cover_imgv) + args.betagans * gan_loss + args.betapix * pixel_loss 


        rev_secret_img = Rnet(container_img)  
        secret_imgv = Variable(secret_img)  
        errR_mse = args.betamse  * mse_loss(rev_secret_img, secret_imgv)
        

        clean_rev_secret_img_A = Rnet(A_imgv)
        clean_imgv = Variable(clean_img)
        errR_clean_A = args.betamse * mse_loss(clean_rev_secret_img_A, clean_imgv)
        clean_rev_secret_img_B = Rnet(cover_imgv)
        clean_imgv = Variable(clean_img)
        errR_clean_B = args.betamse * mse_loss(clean_rev_secret_img_B, clean_imgv)
        errR_clean =args.betacleanA * errR_clean_A + args.betacleanB * errR_clean_B  

        half_batchsize = int(this_batch_size / 2)
        errR_consist = args.betamse *  mse_loss(rev_secret_img[0:half_batchsize, :, : ,:], rev_secret_img[half_batchsize:this_batch_size, : ,: ,:]) 
        
        errR = errR_mse + args.betacons * errR_consist +args.betaclean * errR_clean
        betaerrR_secret = args.beta * errR
        err_sum = errH + betaerrR_secret 


        err_sum.backward()
        optimizerH.step()
        optimizerR.step()


        #  Train Discriminator
        Dnet.zero_grad()
        # Real loss
        pred_real = Dnet(cover_imgv)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = Dnet(container_img.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        errD = 10000 * 0.5 * (loss_real + loss_fake)
        
        errD.backward()
        optimizerD.step()


        Hlosses.update(errH.data, this_batch_size)  
        Rlosses.update(errR.data, this_batch_size) 
        R_mselosses.update(errR_mse.data, this_batch_size) 
        R_consistlosses.update(errR_consist.data, this_batch_size) 

        Dlosses.update(errD.data, this_batch_size)  
        FakeDlosses.update(loss_fake.data, this_batch_size)  
        RealDlosses.update(loss_real.data, this_batch_size)  
        Ganlosses.update(gan_loss.data, this_batch_size)
        Pixellosses.update(pixel_loss.data, this_batch_size) 
        Vgglosses.update(vgg_loss.data, this_batch_size)        
        SumLosses.update(err_sum.data, this_batch_size)

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\tLoss_H: %.4f Loss_R: %.4f Loss_R_mse: %.4f Loss_R_consist: %.4f Loss_D: %.4f Loss_FakeD: %.4f Loss_RealD: %.4f Loss_Gan: %.4f Loss_Pixel: %.4f Loss_Vgg: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, args.epochs, i, len(train_loader),
            Hlosses.val, Rlosses.val, R_mselosses.val, R_consistlosses.val, Dlosses.val, FakeDlosses.val, RealDlosses.val, Ganlosses.val, Pixellosses.val, Vgglosses.val, SumLosses.val, data_time.val, batch_time.val)

        if i % args.logFrequency == 0:
            print_log(log, logPath)
        else:
            print_log(log, logPath, console=False)


        if epoch % 1 == 0 and i % args.resultPicFrequency == 0:
            diff =  50 * (container_img - cover_imgv)
            save_result_pic(this_batch_size, cover_img_A, cover_imgv.data, container_img.data, 
            	secret_img, rev_secret_img.data, clean_rev_secret_img_A.data, clean_rev_secret_img_B.data, diff.data, epoch, i, args.trainpics)

    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH_lr = %.8f      optimizerR_lr = %.8f     optimizerD_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr'], optimizerD.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_Hloss=%.6f\tepoch_Rloss=%.6f\tepoch_R_mseloss=%.6f\tepoch_R_consistloss=%.6f\tepoch_Dloss=%.6f\tepoch_FakeDloss=%.6f\tepoch_RealDloss=%.6f\tepoch_GanLoss=%.6fepoch_Pixelloss=%.6f\tepoch_Vggloss=%.6f\tepoch_sumLoss=%.6f" % (
        Hlosses.avg, Rlosses.avg, R_mselosses.avg, R_consistlosses.avg, Dlosses.avg, FakeDlosses.avg, RealDlosses.avg, Ganlosses.avg, Pixellosses.avg, Vgglosses.avg, SumLosses.avg)


    print_log(epoch_log, logPath)


    writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/D_lr", optimizerD.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/beta", args.beta, epoch)

    writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('train/R_mse_loss', R_mselosses.avg, epoch)
    writer.add_scalar('train/R_consist_loss', R_consistlosses.avg, epoch)    
    writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('train/D_loss', Dlosses.avg, epoch)
    writer.add_scalar('train/FakeD_loss', FakeDlosses.avg, epoch) 
    writer.add_scalar('train/RealD_loss', RealDlosses.avg, epoch)   
    writer.add_scalar('train/Gan_loss', Ganlosses.avg, epoch)  
    writer.add_scalar('train/Pixel_loss', Pixellosses.avg, epoch)
    writer.add_scalar('train/Vgg_loss', Vgglosses.avg, epoch)    
    writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)


def validation(val_loader,  epoch, Hnet, Rnet, Dnet, target_model):
    """Calculate metrics on validation set

    :param val_loader: the validation train loader
    :type val_loader: torch.utils.data.DataLoader
    :param epoch: number of epochs
    :type epoch: int
    :param Hnet: the hiding model; hide a watermark in an image
    :type Hnet: models.HidingUNet
    :param Rnet: the reveal model; reveal a watermark embedded in an image
    :type Rnet: models.HidingRes
    :param Dnet: discriminator model; distinguish between containers (embedded with watermark) and covers (no watermark)
    :type Dnet: models.Discriminator
    :param target_model: The target task model: U-Net reconstruction
    :type target_model: models.Unet
    :return: loss values
    :rtype: list[torch.Tensors]
    """
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Dnet.eval()
    Hlosses = AverageMeter()  
    Rlosses = AverageMeter() 
    R_mselosses = AverageMeter() 
    R_consistlosses = AverageMeter()   
    Dlosses = AverageMeter()  
    FakeDlosses = AverageMeter()
    RealDlosses = AverageMeter()
    Ganlosses = AverageMeter()
    Pixellosses = AverageMeter()
    Vgglosses = AverageMeter()    

    # Tensor type
    Tensor = torch.cuda.FloatTensor 
    with torch.no_grad(): 

        loader = transforms.Compose([transforms.ToTensor(),
                                    trans.Resize((args.imsize, args.imsize), antialias=None)])
        clean_img = Image.open(args.clean)
        clean_img = loader(clean_img)  
        if args.image_signature is None:
            secret_img = Image.new('RGB', (args.imsize, args.imsize), COLORS[args.response_c])
        else:
            secret_img = Image.open(f"src/datasets/signatures/{args.image_signature}.jpg")
        secret_img = loader(secret_img)  
        
        for i, data in enumerate(val_loader, 0):

            Hnet.zero_grad()
            Rnet.zero_grad()
            Dnet.zero_grad()

            if args.gpu:
                data = data.cuda()

            cover_img = target_model(data)
            vgg_loss = mse_loss(cover_img, data)
            
            all_batch_size = int(cover_img.shape[0]) 
            this_batch_size = all_batch_size//2            
            cover_img_A = cover_img[:this_batch_size,:,:,:] # these are clean images that are not watermarked
            cover_img_B = cover_img[this_batch_size:,:,:,:] # these are watermarked images

            secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)
            secret_img = secret_img[0:this_batch_size, :, :, :]  
   
            clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)         
            clean_img = clean_img[0:this_batch_size, :, :, :] 

            if args.gpu:
                cover_img = cover_img.cuda()
                cover_img_A = cover_img_A.cuda()
                cover_img_B = cover_img_B.cuda()             
                secret_img = secret_img.cuda()
                clean_img = clean_img.cuda()

            concat_img = torch.cat([cover_img_B, secret_img], dim=1)
            concat_imgv = Variable(concat_img)  
            cover_imgv = Variable(cover_img_B)  
            container_img = Hnet(concat_imgv)  
            A_imgv = Variable(cover_img_A) 

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((cover_imgv.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((cover_imgv.size(0), *patch))), requires_grad=False)
            pred_fake = Dnet(container_img)
            gan_loss = criterion_GAN(pred_fake, valid) 

            pixel_loss = criterion_pixelwise(container_img, cover_imgv)
            container_img_rgb = container_img.repeat(1,1,1,1)
            cover_imgv_rgb = cover_imgv.repeat(1,1,1,1)
            cover_imgv_rgb.detach()
            
            errH = args.betamse * mse_loss(container_img, cover_imgv) +  args.betagans * gan_loss + args.betapix * pixel_loss

            #  Train Discriminator
            # Real loss
            pred_real = Dnet(cover_imgv)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = Dnet(container_img.detach())
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            errD = 10000 * 0.5 * (loss_real + loss_fake)

            rev_secret_img = Rnet(container_img)  
            secret_imgv = Variable(secret_img) 
            errR_mse = args.betamse * mse_loss(rev_secret_img, secret_imgv)
            
            clean_rev_secret_img_A = Rnet(A_imgv)
            clean_imgv = Variable(clean_img)
            errR_clean_A = args.betamse * mse_loss(clean_rev_secret_img_A, clean_imgv)
            clean_rev_secret_img_B = Rnet(cover_imgv)
            clean_imgv = Variable(clean_img)
            errR_clean_B = args.betamse * mse_loss(clean_rev_secret_img_B, clean_imgv)
            errR_clean =args.betacleanA * errR_clean_A + args.betacleanB * errR_clean_B 

            half_batchsize = int(this_batch_size / 2)
            errR_consist = args.betamse *  mse_loss(rev_secret_img[0:half_batchsize, :, : ,:], rev_secret_img[half_batchsize:half_batchsize * 2, : ,: ,:]) 
            
            errR = errR_mse + args.betacons * errR_consist +args.betaclean * errR_clean
            betaerrR_secret = args.beta * errR
            err_sum = errH + betaerrR_secret 


            Hlosses.update(errH.data, this_batch_size)  
            Rlosses.update(errR.data, this_batch_size) 
            R_mselosses.update(errR_mse.data, this_batch_size)
            R_consistlosses.update(errR_consist.data, this_batch_size)
            Dlosses.update(errD.data, this_batch_size)  
            FakeDlosses.update(loss_fake.data, this_batch_size)  
            RealDlosses.update(loss_real.data, this_batch_size)  
            Ganlosses.update(gan_loss.data, this_batch_size) 
            Pixellosses.update(pixel_loss.data, this_batch_size) 
            Vgglosses.update(vgg_loss.data, this_batch_size) 


            if i % 50 == 0:
                diff =  50 * (container_img - cover_imgv)
                save_result_pic(this_batch_size, cover_img_A, cover_imgv.data, container_img.data, 
                	secret_img, rev_secret_img.data, clean_rev_secret_img_A.data,clean_rev_secret_img_B.data, diff.data, epoch, i, args.validationpics)    

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_r_mseloss = R_mselosses.avg
    val_r_consistloss = R_consistlosses.avg
    val_dloss = Dlosses.avg
    val_fakedloss = FakeDlosses.avg
    val_realdloss = RealDlosses.avg
    val_Ganlosses = Ganlosses.avg
    val_Pixellosses = Pixellosses.avg
    val_Vgglosses = Vgglosses.avg       
    val_sumloss = val_hloss + args.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_R_mseloss = %.6f\t val_R_consistloss = %.6f\t val_Dloss = %.6f\t val_FakeDloss = %.6f\t val_RealDloss = %.6f\t val_Ganlosses = %.6f\t val_Pixellosses = %.6f\t val_Vgglosses = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_r_mseloss, val_r_consistloss, val_dloss, val_fakedloss, val_realdloss, val_Ganlosses, val_Pixellosses, val_Vgglosses, val_sumloss, val_time)

    print_log(val_log, logPath)


    writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
    writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
    writer.add_scalar('validation/R_mse_loss', R_mselosses.avg, epoch)
    writer.add_scalar('validation/R_consist_loss', R_consistlosses.avg, epoch)   
    writer.add_scalar('validation/D_loss_avg', Dlosses.avg, epoch)
    writer.add_scalar('validation/FakeD_loss_avg', FakeDlosses.avg, epoch)
    writer.add_scalar('validation/RealD_loss_avg', RealDlosses.avg, epoch)
    writer.add_scalar('validation/Gan_loss_avg', val_Ganlosses, epoch)
    writer.add_scalar('validation/Pixel_loss_avg', val_Pixellosses, epoch)
    writer.add_scalar('validation/Vgg_loss_avg', val_Vgglosses, epoch)    
    writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)

    print("#################################################### validation end ########################################################")

    return val_hloss, val_rloss, val_r_mseloss, val_r_consistloss, val_dloss, val_fakedloss, val_realdloss, val_Ganlosses, val_Pixellosses, vgg_loss, val_sumloss


def test(val_loader, Hnet, Rnet, Dnet, target_model, save_path):
    """Test models and calculate metrics

    :param val_loader: the validation train loader
    :type val_loader: torch.utils.data.DataLoader
    :param Hnet: the hiding model; hide a watermark in an image
    :type Hnet: models.HidingUNet
    :param Rnet: the reveal model; reveal a watermark embedded in an image
    :type Rnet: models.HidingRes
    :param Dnet: discriminator model; distinguish between containers (embedded with watermark) and covers (no watermark)
    :type Dnet: models.Discriminator
    :param target_model: The target task model: U-Net reconstruction
    :type target_model: models.Unet
    :param save_path: where to save images and test results
    :type save_path: str
    :return: test metrics
    :rtype: dict
    """
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Dnet.eval()

    test_outputs = defaultdict(list)
    tot_mse_recon = 0
    tot_psnr_recon = 0
    tot_ssim_recon = 0
    tot_mse_water = 0
    tot_psnr_water = 0
    tot_ssim_water = 0

    # Tensor type
    Tensor = torch.cuda.FloatTensor 
    with torch.no_grad(): 
        # -------------------------------
        # Load secret and clean image
        # -------------------------------
        loader = transforms.Compose([transforms.ToTensor(),
                                    trans.Resize((args.imsize, args.imsize), antialias=None)])
        clean_img = Image.open(args.clean)
        clean_img = loader(clean_img)  
        if args.image_signature is None:
            secret_img = Image.new('RGB', (args.imsize, args.imsize), COLORS[args.response_c])
        else:
            secret_img = Image.open(f"src/datasets/signatures/{args.image_signature}.jpg")
        secret_img = loader(secret_img)  
        # -------------------------------
        # Testing :)
        # -------------------------------
        first_batch = True
        for i, data in enumerate(val_loader, 0):

            Hnet.zero_grad()
            Rnet.zero_grad()
            Dnet.zero_grad()

            if args.gpu:
                data = data.cuda()

            cover_img = target_model(data)
            
            all_batch_size = int(cover_img.shape[0]) 
            this_batch_size = all_batch_size//2
            secret_img = secret_img.repeat(all_batch_size, 1, 1, 1)
            secret_img = secret_img[0:all_batch_size, :, :, :]  

            if args.gpu:
                cover_img = cover_img.cuda()            
                secret_img = secret_img.cuda()

            concat_img = torch.cat([cover_img, secret_img], dim=1)
            concat_imgv = Variable(concat_img)  
            # ----------------------------
            # Make containers
            # ----------------------------
            container_img = Hnet(concat_imgv)
            # ----------------------------
            # Get revealed secret
            # ----------------------------
            rev_watermark_img = Rnet(container_img)
            rev_clean_img = Rnet(cover_img)
            # ----------------------------
            # Calculate Metrics
            # ----------------------------
            recon_metrics = calculate_image_metrics(cover_img, data)                  # Recon model performance MSE(data, cover_img)
            hide_metrics = calculate_image_metrics(container_img, cover_img)          # Hide model performance MSE(cover, container)
            rev_clean_metrics = calculate_image_metrics(rev_clean_img, secret_img)    # Reveal model performance on NOT watermarked ---> MSE should be high
            rev_water_metrics = calculate_image_metrics(rev_watermark_img, secret_img)# Reveal model performance on WATERMARKED  ---> MSE should be low
            #
            # Save images if first batch
            #
            if first_batch:
                os.makedirs(save_path, exist_ok=True)
                # get random images
                indices = torch.randperm(cover_img.shape[0])[:args.num_test_save]
                # save reconstruction images
                for i, images in enumerate(zip(data[indices], cover_img[indices])):
                    in_im, out_im = images
                    save_image(in_im, os.path.join(save_path, f'input_recon_{i}.jpg'))
                    save_image(out_im, os.path.join(save_path, f'output_recon_{i}.jpg'))
                # save watermark with non-watermarked image
                for i, images in enumerate(zip(cover_img[indices], rev_clean_img[indices])):
                    in_im, out_im = images
                    save_image(in_im, os.path.join(save_path, f'input_benign_{i}.jpg'))
                    save_image(out_im, os.path.join(save_path, f'output_benign_{i}.jpg'))
                # save watermark results
                for i, images in enumerate(zip(container_img[indices], rev_watermark_img[indices], secret_img[indices])):
                    in_im, out_im, sec_im = images
                    save_image(in_im, os.path.join(save_path, f'input_trigger_{i}.jpg'))
                    save_image(out_im, os.path.join(save_path, f'output_trigger_{i}.jpg'))
                    save_image(sec_im, os.path.join(save_path, f'secret_{i}.jpg'))
 
                first_batch = False

            for key, value in rev_water_metrics.items():
                test_outputs[f"{key}_trigger"].append(value.mean())
            for key, value in rev_clean_metrics.items():
                test_outputs[f"{key}_benign"].append(value.mean())
            for key, value in recon_metrics.items():
                test_outputs[f"{key}_recon"].append(value.mean())
            for key, value in hide_metrics.items():
                test_outputs[f"{key}_hide"].append(value.mean())

        avg_metrics = {metric: torch.stack(test_outputs[metric]).mean() for metric in test_outputs}
        for key, value in avg_metrics.items():
              avg_metrics[key] = value.cpu().item()

    return avg_metrics



def weights_init(m):
    """Custom weights initialization called on netG and netD

    :param m: the model to initalize
    :type m: Rnet, Dnet, Hnet
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_network(net):
    """Print the structure and parameters number of the net

    :param net: the network to print
    :type net: Rnet, Dnet, Hnet
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


def save_current_codes(des_path):
    """Save current codes to a specified path.

    :param des_path: where to save the current codes
    :type des_path: str
    """
    main_file_path = os.path.realpath(__file__)  
    cur_work_dir, mainfile = os.path.split(main_file_path)

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


def print_log(log_info, log_path, console=True):
    """Print the training log and save into logFiles

    :param log_info: info logged during training/validation
    :type log_info: dict
    :param log_path: where to save the log info
    :type log_path: str
    :param console: whether to show the console
    :type console: bool
    """
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not args.debug:
        # write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


def save_result_pic(this_batch_size, originalLabelvA, originalLabelvB, Container_allImg, secretLabelv, RevSecImg,RevCleanImgA,RevCleanImgB, diff, epoch, i, save_path):
    """Save result pic and the coverImg filePath and the secretImg filePath

    :param this_batch_size: current batch size (number of images to save)
    :type this_batch_size: int
    :param originalLabelvA: Original label of input set A
    :type originalLabelvA: torch.Tensor
    :param originalLabelvB: Original label of input set B
    :type originalLabelvB: torch.Tensor
    :param Container_allImg: Container images created from all input images
    :type Container_allImg: torch.Tensor
    :param secretLabelv: Secrets embedded in all input images to create a container
    :type secretLabelv: torch.Tensor
    :param RevSecImg: Reveal images from containers
    :type RevSecImg: torch.Tensor
    :param RevCleanImgA: Reveal images from clean set A
    :type RevCleanImgA: torch.Tensor
    :param RevCleanImgB: Reveal images from clean set B
    :type RevCleanImgB: torch.Tensor
    :param diff: The difference between RevSecImg and Secret
    :type diff: torch.Tensor
    :param epoch: the current epoch of training or validation
    :type epoch: int
    :param i: the current batch number
    :type i: int
    :param save_path: where to save the images
    :type save_path: str
    """
    originalFramesA = originalLabelvA.resize_(this_batch_size, 3, args.imsize, args.imsize)
    originalFramesB = originalLabelvB.resize_(this_batch_size, 3, args.imsize, args.imsize)
    container_allFrames = Container_allImg.resize_(this_batch_size, 3, args.imsize, args.imsize)

    secretFrames = secretLabelv.resize_(this_batch_size, 3, args.imsize, args.imsize)
    revSecFrames = RevSecImg.resize_(this_batch_size, 3, args.imsize, args.imsize)
    revCleanFramesA = RevCleanImgA.resize_(this_batch_size, 3, args.imsize, args.imsize)
    revCleanFramesB = RevCleanImgB.resize_(this_batch_size, 3, args.imsize, args.imsize)

    showResult = torch.cat([secretFrames,originalFramesA,revCleanFramesA, originalFramesB,revCleanFramesB, diff, container_allFrames,
            revSecFrames,], 0)
    
    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    vutils.save_image(showResult, resultImgName, nrow=this_batch_size, padding=1, normalize=False)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()

