#! /bin/bash

MODEL="fpn"
BACKBONE="mobilenet_v2"
DATASET="optic"
DATA_PATH="RIM-ONE_DL"
ATTACK="ftune1"
EXP="rq5_green"

echo "${EXP}: ${MODEL} ${DATASET} Semseg -- Attack ${ATTACK}"
python src/train_pylight_semseg.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "${EXP}_${MODEL}_semseg_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ATTACK="ftune5"

echo "${EXP}: ${MODEL} ${DATASET} Semseg -- Attack ${ATTACK}"
python src/train_pylight_semseg.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "${EXP}_${MODEL}_semseg_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ATTACK="overwrite"

echo "${EXP}: ${MODEL} ${DATASET} Semseg -- Attack ${ATTACK}"
python src/train_pylight_semseg.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "${EXP}_${MODEL}_semseg_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}