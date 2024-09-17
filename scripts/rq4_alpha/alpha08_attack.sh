#! /bin/bash

MODEL="unet"
BACKBONE="mobilenet_v2"
DATASET="imagenet"
DATA_PATH="ImageNet"
EXP="rq4"
ALPHA=0.8
ATTACK="ftune1"

echo "${EXP}: ${MODEL} ${DATASET} (alpha --> ${ALPHA}) ATTACK ${ATTACK}........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/alpha_${ALPHA}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/alpha_${ALPHA}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-small" \
    --metrics_fname "${EXP}_${MODEL}_alpha=${ALPHA}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ATTACK="ftune5"

echo "${EXP}: ${MODEL} ${DATASET} (alpha --> ${ALPHA}) ATTACK ${ATTACK}........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/alpha_${ALPHA}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/alpha_${ALPHA}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-small" \
    --metrics_fname "${EXP}_${MODEL}_alpha=${ALPHA}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ATTACK="overwrite"

echo "${EXP}: ${MODEL} ${DATASET} (alpha --> ${ALPHA}) ATTACK ${ATTACK}........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/alpha_${ALPHA}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/alpha_${ALPHA}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-small" \
    --metrics_fname "${EXP}_${MODEL}_alpha=${ALPHA}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}