#! /bin/bash

MODEL="fpn"
BACKBONE="mobilenet_v2"
DATASET="cifar"
DATASET_PATH=None

echo "Expr 1 Fragile ${DATASET} ATTACK: Finetune 1 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/attack/${DATASET}_ftune1" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "rq1_${MODEL}_${DATASET}_attack_ftune1" \
    --attack ftune1 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATASET_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

echo "Expr 1 Fragile ImageNet ATTACK: Finetune 5 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/attack/${DATASET}_ftune5" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "rq1_${MODEL}_${DATASET}_attack_ftune5" \
    --attack ftune5 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATASET_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

echo "Expr 1 Fragile ImageNet ATTACK: Overwrite ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/attack/${DATASET}_overwrite" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "rq1_${MODEL}_${DATASET}_attack_overwrite" \
    --image_signature flower \
    --attack overwrite \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATASET_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}