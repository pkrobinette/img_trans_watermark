#! /bin/bash

echo "Expr 1 lraspp CIFAR-10 TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/lraspp/train/cifar' \
    --metrics_fname 'rq1_lraspp_cifar_train' \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar \
    --model_type 'lraspp' \
    --backbone 'mobilenet'

echo "Expr 1 lraspp ImageNet TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/lraspp/train/imagenet' \
    --metrics_fname 'rq1_lraspp_imagenet_train' \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet \
    --model_type 'lraspp' \
    --backbone 'mobilenet'

echo "Expr 1 lraspp CLWD TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'drive/MyDrive/SatML_watermarking/rq1/lraspp/train/clwd' \
    --metrics_fname 'rq1_lraspp_clwd_train' \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset clwd \
    --data_path CLWD \
    --model_type 'lraspp' \
    --backbone 'mobilenet'