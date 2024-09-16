"""
Training file to train a Unet Classification model ;>

"""

from modules.module_class_watermark import UNetClassModule
from models.Unet_class import UNetClass
import lightning.pytorch as pl
import os
import argparse
import torch
import pickle
from utils import load_data, get_name_class, pretty_print_args, load_class_data,\
    prune_model_global_unstructured
import glob
import sys


def get_args():
    """Get user arguments

    :return: the user arguments
    :rtype: Argparse
    """
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--num_devices', type=int, default=None, help='Number of devices to use for training')
    parser.add_argument('--gpu', action='store_true', help='Flag to use GPU if available')
    parser.add_argument('--save_path', type=str, help='name to save file')
    parser.add_argument("--trigger_c", type=str, default="purple", help="color of trigger")
    parser.add_argument("--trigger_s", type=str, default="small", help="size of the trigger")
    parser.add_argument("--trigger_pos", type=str, default="top_right", help="position of trigger")
    parser.add_argument("--num_test_save", type=int, default=10, help="number of test images to save; will be 2x (normal and triggers)")
    parser.add_argument("--no_trigger", action='store_true', help='flag to train without triggers')
    parser.add_argument("--checkpoint", type=str, default=None, help="training checkpoint to finetune")
    parser.add_argument("--test", action="store_true", help="just testing model")
    parser.add_argument("--dataset", type=str, default="cifar10", help='dataset used for training')
    parser.add_argument("--data_path", type=str, default=None, help='path of dataset')
    parser.add_argument("--noise_trigger", action="store_true", help="noise trigger")
    parser.add_argument("--steg_trigger", action='store_true', help="steg_signature")
    parser.add_argument("--attack", type=str, default=None, help="model attack: [prune10, prune40, ftune1, ftune5]")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight of watermarking loss during training")
    #
    # Parse the arguments
    #
    args = parser.parse_args()
    
    return args
    

def main():
    """ The main function """
    args = get_args()    
    assert args.dataset in ['cifar10']
    pretty_print_args(args)
    model_name, expr_name = get_name_class(args)
    expr_path = os.path.join(args.save_path, expr_name)
    #
    # Init model
    #
    print("Initializing models ...")
    unet = UNetClass(
        im_size=128,
        backbone="mobilenet_v2", 
        c_in=3,
        classes=2
    )
    model = UNetClassModule(
        model=unet,      # model
        alpha=args.alpha, # weight of watermarking loss
        lr=0.001,        # learning rate
        num_images_save=args.num_test_save, # number of images to save during testing
        save_path=os.path.join(expr_path, "images")
    )
    #
    # Load checkpoint of indicated for finetuning
    #
    if args.checkpoint is not None:
        print("Loading pretrained checkpoint ... ")
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint, "**.pth")))
        checkpoint = torch.load(checkpoints[-1]) # take the most recent
        model_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.model.load_state_dict(model_state_dict)
    # -----------------------
    # Model ATTACKS
    # -----------------------
    ftune_attack = False
    if args.attack:
        if args.attack == "prune10":
            print("Pruning 10% of model ...")
            prune_model_global_unstructured(model, pruning_ratio=0.1)
            args.test = True
        elif args.attack == "prune40":
            print("Pruning 40% of model ...")
            prune_model_global_unstructured(model, pruning_ratio=0.4)
            args.test = True
        elif args.attack == "ftune1":
            args.epochs = 1
            ftune_attack = True
            args.no_trigger = True
            args.test = False
        elif args.attack == "ftune5":
            args.epochs = 5
            ftune_attack = True
            args.no_trigger = True
            args.test = False

    #
    # If training and test
    # ftune = True, test= False
    #
    if not args.test or ftune_attack==True:
        os.makedirs(expr_path, exist_ok=True)
        #
        # Dataset
        #
        train_loader = load_class_data("train", args)
        #
        # Set up trainer
        #
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            devices=args.num_devices,
            accelerator="gpu" if args.gpu else "cpu",
        )
        #
        # Train
        #
        print("Training ...")
        trainer.fit(model=model, train_dataloaders=train_loader)
        #
        # Save model
        #
        print("Model saved to: ", os.path.join(expr_path, model_name))
        if not args.attack:
            torch.save(model.model.state_dict(), os.path.join(expr_path, model_name))

    if args.attack is not None:
        os.makedirs(expr_path, exist_ok=True)
        torch.save(model, os.path.join(expr_path, model_name))
    #
    # Test dataset
    # All test should be on triggers
    #
    args.no_trigger = False
    test_loader = load_class_data("test", args)
    #
    # Test
    #
    model.eval()
    trainer = pl.Trainer(
        max_epochs=1,
        devices=args.num_devices,
        accelerator="gpu" if args.gpu else "cpu",
    )
    
    metrics = trainer.test(model, dataloaders=test_loader)

    with open(os.path.join(expr_path, f'test_results.pkl'), 'wb') as file:
        pickle.dump(metrics[0], file)
    

if __name__ == "__main__":
    main()

