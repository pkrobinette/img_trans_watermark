#! /bin/bash


echo "Expr 1 Fragile CIFAR-10 TEST ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/fragile/train/fragile_cifar_test' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/fragile/train/fragile_cifar_train/unet_pu-tl-small_gr-tr-small' \
    --metrics_fname 'exp1_fragile_cifar_train' \
    --image_signature ECCV \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar10

echo "Expr 1 Fragile CIFAR-10 ATTACK: Finetune 1 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/fragile/attack/fragile_cifar_attack_ftune1' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/fragile/train/fragile_cifar_train/unet_pu-tl-small_gr-tr-small' \
    --metrics_fname 'exp1_fragile_cifar_attack_ftune1' \
    --image_signature ECCV \
    --attack ftune1 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar10

echo "Expr 1 Fragile CIFAR-10 ATTACK: Finetune 5 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/fragile/attack/fragile_cifar_attack_ftune5' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/fragile/train/fragile_cifar_train/unet_pu-tl-small_gr-tr-small' \
    --metrics_fname 'exp1_fragile_cifar_attack_ftune5' \
    --image_signature ECCV \
    --attack ftune5 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar10

echo "Expr 1 Fragile CIFAR-10 ATTACK: Overwrite ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c green \
    --trigger_pos center \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/fragile/attack/fragile_cifar_attack_overwrite' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/fragile/train/fragile_cifar_train/unet_pu-tl-small_gr-tr-small' \
    --metrics_fname 'exp1_fragile_cifar_attack_overwrite' \
    --image_signature flower \
    --attack overwrite \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar10