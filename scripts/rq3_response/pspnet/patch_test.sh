#! /bin/bash

MODEL="pspnet"
BACKBONE="mobilenet_v2"
DATASET="imagenet"
DATA_PATH="ImageNet"
EXP="rq3"
RESPONSE="patch_patch"

echo "${EXP}: ${MODEL} ${DATASET} (${RESPONSE}) ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c pink \
    --response_s quarter \
    --response_pos bottom_right \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/test/${DATASET}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/train/${DATASET}/${MODEL}_pu-tl-small_pi-br-quarter" \
    --metrics_fname "${EXP}_${MODEL}_${RESPONSE}_${DATASET}_test" \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}