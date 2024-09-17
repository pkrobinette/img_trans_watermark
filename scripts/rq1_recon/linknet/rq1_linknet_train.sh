#! /bin/bash

echo "Expr 1 linknet CIFAR-10 TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/linknet/train/cifar' \
    --metrics_fname 'rq1_linknet_cifar_train' \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar \
    --model_type 'linknet' \
    --backbone 'mobilenet_v2'

echo "Expr 1 linknet ImageNet TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/linknet/train/imagenet' \
    --metrics_fname 'rq1_linknet_imagenet_train' \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet \
    --model_type 'linknet' \
    --backbone 'mobilenet_v2'

echo "Expr 1 linknet CLWD TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/linknet/train/clwd' \
    --metrics_fname 'rq1_linknet_clwd_train' \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset clwd \
    --data_path CLWD \
    --model_type 'linknet' \
    --backbone 'mobilenet_v2'