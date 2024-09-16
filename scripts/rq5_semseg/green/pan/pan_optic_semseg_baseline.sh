#! /bin/bash

MODEL="pan"
BACKBONE="mobilenet_v2"
DATASET="optic"
DATA_PATH="RIM-ONE_DL"
EXP="rq5_green"

echo "${EXP}: ${MODEL} ${DATASET} Semseg"
python src/train_pylight_semseg.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/baseline/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_semseg_${DATASET}_baseline" \
    --epochs 200 \
    --alpha 0.0 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}