#! /bin/bash

MODEL="deeplabv3"
BACKBONE="resnet"
DATASET="imagenet"
DATA_PATH="ImageNet"
EXP="rq2"
TRIGGER="block"

echo "${EXP}: ${MODEL} ${DATASET} (${TRIGGER}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s full \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER}/train/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_${TRIGGER}_${DATASET}_train" \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

TRIGGER="noise"

echo "${EXP}: ${MODEL} ${DATASET} (${TRIGGER}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s full \
    --noise_trigger \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER}/train/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_${TRIGGER}_${DATASET}_train" \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

TRIGGER="steg"

echo "${EXP}: ${MODEL} ${DATASET} (${TRIGGER}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s full \
    --steg_trigger \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER}/train/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_${TRIGGER}_${DATASET}_train" \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}