#! /bin/bash

MODEL="pspnet"
BACKBONE="mobilenet_v2"
DATASET="imagenet"
DATA_PATH="ImageNet"
EXP="rq3"
RESPONSE="patch_image"
ATTACK="ftune1"

echo "${EXP}: ${MODEL} ${DATASET} (${RESPONSE}) ATTACK ${ATTACK}........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-small" \
    --metrics_fname "${EXP}_${MODEL}_${RESPONSE}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ATTACK="ftune5"

echo "${EXP}: ${MODEL} ${DATASET} (${RESPONSE}) ATTACK ${ATTACK}........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-small" \
    --metrics_fname "${EXP}_${MODEL}_${RESPONSE}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ATTACK="overwrite"

echo "${EXP}: ${MODEL} ${DATASET} (${RESPONSE}) ATTACK ${ATTACK}........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-small" \
    --metrics_fname "${EXP}_${MODEL}_${RESPONSE}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}