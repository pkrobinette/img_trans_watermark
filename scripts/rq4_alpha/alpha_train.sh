#! /bin/bash

MODEL="unet"
BACKBONE="mobilenet_v2"
DATASET="imagenet"
DATA_PATH="ImageNet"
EXP="rq4"
ALPHA=0.8

echo "${EXP}: ${MODEL} ${DATASET} (alpha --> ${ALPHA}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/alpha_${ALPHA}/train/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_alpha=${ALPHA}_${DATASET}_train" \
    --alpha ${ALPHA} \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ALPHA=0.6

echo "${EXP}: ${MODEL} ${DATASET} (alpha --> ${ALPHA}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/alpha_${ALPHA}/train/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_alpha=${ALPHA}_${DATASET}_train" \
    --alpha ${ALPHA} \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ALPHA=0.4

echo "${EXP}: ${MODEL} ${DATASET} (alpha --> ${ALPHA}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/alpha_${ALPHA}/train/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_alpha=${ALPHA}_${DATASET}_train" \
    --alpha ${ALPHA} \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ALPHA=0.2

echo "${EXP}: ${MODEL} ${DATASET} (alpha --> ${ALPHA}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/alpha_${ALPHA}/train/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_alpha=${ALPHA}_${DATASET}_train" \
    --alpha ${ALPHA} \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}