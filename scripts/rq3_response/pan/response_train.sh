#! /bin/bash

MODEL="pan"
BACKBONE="mobilenet_v2"
DATASET="imagenet"
DATA_PATH="ImageNet"
EXP="rq3"
RESPONSE="patch_patch"

echo "${EXP}: ${MODEL} ${DATASET} (${RESPONSE}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c pink \
    --response_s quarter \
    --response_pos bottom_right \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/train/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_${RESPONSE}_${DATASET}_train" \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

RESPONSE="patch_image"

echo "${EXP}: ${MODEL} ${DATASET} (${RESPONSE}) TRAIN ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/train/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_${RESPONSE}_${DATASET}_train" \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}
