"""
Training file to train a Unet semseg model ;>

"""

from modules.module_semseg_watermark import SemsegModule
import lightning.pytorch as pl
import os
import argparse
import torch
import pickle
import utils
import glob


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
    parser.add_argument("--response_c", type=str, default="green", help="color or signature")
    parser.add_argument("--response_s", type=str, default="small", help="size of the signature")
    parser.add_argument("--trigger_pos", type=str, default="top_right", help="position of trigger")
    parser.add_argument("--response_pos", type=str, default="top_right", help="position of signature")
    parser.add_argument("--num_test_save", type=int, default=10, help="number of test images to save; will be 2x (normal and triggers)")
    parser.add_argument("--no_trigger", action='store_true', help='flag to train without triggers')
    parser.add_argument("--checkpoint", type=str, default=None, help="training checkpoint to finetune")
    parser.add_argument("--test", action="store_true", help="just testing model")
    parser.add_argument("--dataset", type=str, default="cifar", help='dataset used for training')
    parser.add_argument("--data_path", type=str, default=None, help='path of dataset')
    parser.add_argument("--noise_trigger", action="store_true", help="noise trigger")
    parser.add_argument("--image_signature", type=str, default=None, help="image_signature")
    parser.add_argument("--steg_trigger", action='store_true', help="steg_signature")
    parser.add_argument("--attack", type=str, default=None, help="attack type")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight of watermarking loss during training")
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--imsize', type=int, default=128, help='the number of frames')
    parser.add_argument('--metrics_fname', type=str, default='test_results', help='the name of the test results file')
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to use from dataset')
    parser.add_argument("--model_type", type=str, default=None, help="The model type to use.")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2", help="The backbone to use for the model.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes in the semantic segmentation dataset.")
    #
    # Parse the arguments
    #
    args = parser.parse_args()
    
    return args
    

def main():
    """ The main function """
    args = get_args()    
    assert args.dataset in ['coco', 'optic']
    utils.pretty_print_args(args)
    model_name, expr_name = utils.get_name(args)
    expr_path = os.path.join(args.save_path, expr_name)
    #
    # Init model
    #
    print("Initializing models ...")
    model = utils.get_semseg_model(model_type=args.model_type, backbone_type=args.backbone, num_classes=args.num_classes)
    model = SemsegModule(
        model=model,      # model
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
            print("ATTACK: Pruning 10% of model .....................................................")
            utils.prune_model_global_unstructured(model, pruning_ratio=0.1)
            args.test = True
        elif args.attack == "prune40":
            print("ATTACK: Pruning 40% of model ........................................................")
            utils.prune_model_global_unstructured(model, pruning_ratio=0.4)
            args.test = True
        elif args.attack == "ftune1":
            print("ATTACK: Finetuning 1 epoch ........................................................")
            args.epochs = 1
            ftune_attack = True
            args.no_trigger = True
            args.test = False
        elif args.attack == "ftune5":
            print("ATTACK: Finetuning 5 epochs ........................................................")
            args.epochs = 5
            ftune_attack = True
            args.no_trigger = True
            args.test = False
        elif args.attack == "overwrite":
            # the overwrite args when running should be those used in training.
            # to change overwrite attack, see below.
            print("ATTACK: Overwriting watermark ........................................................")
            args.epochs=5
            temp_trigger_c = args.trigger_c
            temp_trigger_s = args.trigger_s
            temp_trigger_pos = args.trigger_pos
            temp_noise_trigger = args.noise_trigger
            temp_steg_trigger = args.steg_trigger
            temp_image_signature = args.image_signature
            temp_response_c = args.response_c
            temp_response_s = args.response_s
            temp_response_pos = args.response_pos
            args.trigger_c = "green"
            args.trigger_s = "small"
            args.trigger_pos = "center"
            args.image_signature = "squirrel"
    #
    # If training and test
    #
    if not args.test or ftune_attack==True:
        os.makedirs(expr_path, exist_ok=True)
        #
        # Dataset
        #
        train_loader = utils.load_semseg_data("train", args)
        #
        # Set up trainer
        #
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            devices=args.num_devices,
            accelerator="gpu" if args.gpu else None,
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
        torch.save(model.model, os.path.join(expr_path, model_name))
    #
    # Test dataset
    # All test should be on triggers
    #
    if args.attack == "overwrite":
        args.image_signature = temp_image_signature
        args.response_c = temp_response_c
        args.response_s = temp_response_s
        args.response_pos = temp_response_pos
        args.trigger_c = temp_trigger_c
        args.trigger_s = temp_trigger_s
        args.trigger_pos = temp_trigger_pos
        args.noise_trigger = temp_noise_trigger
        args.steg_trigger = temp_steg_trigger
    args.no_trigger = False
    test_loader = utils.load_semseg_data("test", args)
    #
    # Test
    #
    model.eval()
    trainer = pl.Trainer(
        max_epochs=1,
        devices=args.num_devices,
        accelerator="gpu" if args.gpu else None,
    )
    
    metrics = trainer.test(model, dataloaders=test_loader)

    with open(os.path.join(expr_path, f'{args.metrics_fname}.pkl'), 'wb') as file:
        pickle.dump(metrics[0], file)
    

if __name__ == "__main__":
    main()
