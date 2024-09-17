#! /bin/bash

echo "Expr 1 Fragile ImageNet ATTACK: Finetune 1 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/fragile/train/fragile_imagenet_test' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/fragile/train/fragile_imagenet_train/unet_pu-tl-small_gr-tr-small' \
    --metrics_fname 'exp1_fragile_imagenet_train' \
    --image_signature ECCV \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet

echo "Expr 1 Fragile ImageNet ATTACK: Finetune 1 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/fragile/attack/fragile_imagenet_attack_ftune1_test' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/fragile/attack/fragile_imagenet_attack_ftune1/unet_pu-tl-small_gr-tr-small' \
    --metrics_fname 'exp1_fragile_imagenet_attack_ftune1' \
    --image_signature ECCV \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet

echo "Expr 1 Fragile ImageNet ATTACK: Finetune 5 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/fragile/attack/fragile_imagenet_attack_ftune5_test' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/fragile/attack/fragile_imagenet_attack_ftune5/unet_pu-tl-small_gr-tr-small' \
    --metrics_fname 'exp1_fragile_imagenet_attack_ftune5' \
    --image_signature ECCV \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet

echo "Expr 1 Fragile ImageNet ATTACK: Overwrite ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c green \
    --trigger_pos center \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/fragile/attack/fragile_imagenet_attack_overwrite_test' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/fragile/attack/fragile_imagenet_attack_overwrite/unet_gr-ce-small_gr-tr-small' \
    --metrics_fname 'exp1_fragile_imagenet_attack_overwrite' \
    --image_signature ECCV \
    --test \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet