#! /bin/bash

EXP_NAME="baseline"
MODEL="pspnet"
BACKBONE="mobilenet_v2"

echo "${EXP_NAME} ${MODEL} CIFAR-10 TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP_NAME}/${MODEL}/train/cifar" \
    --metrics_fname "${EXP_NAME}_${MODEL}_cifar_train" \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset cifar \
    --model_type ${MODEL} \
    --backbone ${BACKBONE} \
    --alpha 0

echo "${EXP_NAME} ${MODEL} ImageNet TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP_NAME}/${MODEL}/train/imagenet" \
    --metrics_fname "${EXP_NAME}_${MODEL}_imagenet_train" \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset imagenet \
    --data_path ImageNet \
    --model_type ${MODEL} \
    --backbone ${BACKBONE} \
    --alpha 0

echo "${EXP_NAME} ${MODEL} CLWD TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP_NAME}/${MODEL}/train/clwd" \
    --metrics_fname "${EXP_NAME}_${MODEL}_clwd_train" \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset clwd \
    --data_path CLWD \
    --model_type ${MODEL} \
    --backbone ${BACKBONE} \
    --alpha 0