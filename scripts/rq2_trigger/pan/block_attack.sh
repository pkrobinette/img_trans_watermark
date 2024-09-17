#! /bin/bash

MODEL="pan"
BACKBONE="mobilenet_v2"
DATASET="imagenet"
DATA_PATH="ImageNet"
EXP="rq2"
TRIGGER="block"
ATTACK="ftune1"

echo "${EXP}: ${MODEL} ${DATASET} (${TRIGGER}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s full \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER}/train/${DATASET}/${MODEL}_pu-tl-full_gr-tr-full" \
    --metrics_fname "${EXP}_${MODEL}_${TRIGGER}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ATTACK="ftune5"

echo "${EXP}: ${MODEL} ${DATASET} (${TRIGGER}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s full \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER}/train/${DATASET}/${MODEL}_pu-tl-full_gr-tr-full" \
    --metrics_fname "${EXP}_${MODEL}_${TRIGGER}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ATTACK="overwrite"

echo "${EXP}: ${MODEL} ${DATASET} (${TRIGGER}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s full \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER}/train/${DATASET}/${MODEL}_pu-tl-full_gr-tr-full" \
    --metrics_fname "${EXP}_${MODEL}_${TRIGGER}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}