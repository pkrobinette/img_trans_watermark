#! /bin/bash

echo "Expr 1 Post-Baseline CIFAR-10 TRAIN ........................................"
# Trigger color - Purple
python src/train_spatial_post_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/cifar' \
    --metrics_fname 'rq1_postbaseline_cifar_train' \
    --epochs 5 \
    --epochs_target_model 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar


echo "Expr 1 Post-Baseline ImageNet TRAIN ........................................"
# Trigger color - Purple
python src/train_spatial_post_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/imagenet' \
    --metrics_fname 'rq1_postbaseline_imagenet_train' \
    --epochs 5 \
    --epochs_target_model 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet

echo "Expr 1 Post-Baseline CLWD TRAIN ........................................"
# Trigger color - Purple
python src/train_spatial_post_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/clwd' \
    --metrics_fname 'rq1_postbaseline_clwd_train' \
    --epochs 5 \
    --epochs_target_model 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset clwd \
    --data_path CLWD