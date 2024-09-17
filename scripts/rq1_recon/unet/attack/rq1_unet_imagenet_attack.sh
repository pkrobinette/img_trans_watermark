#! /bin/bash

echo "Expr 1 Fragile ImageNet ATTACK: Finetune 1 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/unet/attack/imagenet_ftune1' \
    --checkpoint 'drive/MyDrive/SatML_watermarking/rq1/unet/train/imagenet/unet_pu-tl-small_gr-tr-full' \
    --metrics_fname 'rq1_unet_imagenet_attack_ftune1' \
    --attack ftune1 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet \
    --model_type 'unet' \
    --backbone 'mobilenet_v2'

echo "Expr 1 Fragile ImageNet ATTACK: Finetune 5 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/unet/attack/imagenet_ftune5' \
    --checkpoint 'drive/MyDrive/SatML_watermarking/rq1/unet/train/imagenet/unet_pu-tl-small_gr-tr-full' \
    --metrics_fname 'rq1_unet_imagenet_attack_ftune5' \
    --attack ftune5 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet \
    --model_type 'unet' \
    --backbone 'mobilenet_v2'

echo "Expr 1 Fragile ImageNet ATTACK: Overwrite ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/unet/attack/imagenet_overwrite' \
    --checkpoint 'drive/MyDrive/SatML_watermarking/rq1/unet/train/imagenet/unet_pu-tl-small_gr-tr-full' \
    --metrics_fname 'rq1_unet_imagenet_attack_overwrite' \
    --image_signature flower \
    --attack overwrite \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet \
    --model_type 'unet' \
    --backbone 'mobilenet_v2'