#! /bin/bash

echo "Expr 1 Fragile CIFAR-10 ATTACK: Finetune 1 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/unet/attack/cifar_ftune1' \
    --checkpoint 'drive/MyDrive/SatML_watermarking/rq1/unet/train/cifar/unet_pu-tl-small_gr-tr-full' \
    --metrics_fname 'rq1_unet_cifar_attack_ftune1' \
    --attack ftune1 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar10 \
    --model_type 'unet' \
    --backbone 'mobilenet_v2'

echo "Expr 1 Fragile CIFAR-10 ATTACK: Finetune 5 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/unet/attack/cifar_ftune5' \
    --checkpoint 'drive/MyDrive/SatML_watermarking/rq1/unet/train/cifar/unet_pu-tl-small_gr-tr-full' \
    --metrics_fname 'rq1_unet_cifar_attack_ftune5' \
    --attack ftune5 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar10 \
    --model_type 'unet' \
    --backbone 'mobilenet_v2'

echo "Expr 1 Fragile CIFAR-10 ATTACK: Overwrite ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/unet/attack/cifar_overwrite' \
    --checkpoint 'drive/MyDrive/SatML_watermarking/rq1/unet/train/cifar/unet_pu-tl-small_gr-tr-full' \
    --metrics_fname 'rq1_unet_cifar_attack_overwrite' \
    --image_signature flower \
    --attack overwrite \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar10 \
    --model_type 'unet' \
    --backbone 'mobilenet_v2'