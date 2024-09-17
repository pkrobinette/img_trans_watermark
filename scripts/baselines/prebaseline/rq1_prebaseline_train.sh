#! /bin/bash

echo "Expr 1 Pre-Baseline CIFAR-10 TRAIN ........................................"
# Trigger color - Purple
python src/train_spatial_pre_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/prebaseline/train/cifar' \
    --metrics_fname 'rq1_prebaseline_cifar_train' \
    --epochs 5 \
    --epochs_target_model 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar10


echo "Expr 1 Pre-Baseline ImageNet TRAIN ........................................"
# Trigger color - Purple
python src/train_spatial_pre_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/prebaseline/train/imagenet' \
    --metrics_fname 'rq1_prebaseline_imagenet_train' \
    --epochs 5 \
    --epochs_target_model 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet

echo "Expr 1 Pre-Baseline CLWD TRAIN ........................................"
# Trigger color - Purple
python src/train_spatial_pre_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/prebaseline/train/clwd' \
    --metrics_fname 'rq1_prebaseline_clwd_train' \
    --epochs 5 \
    --epochs_target_model 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset clwd \
    --data_path CLWD