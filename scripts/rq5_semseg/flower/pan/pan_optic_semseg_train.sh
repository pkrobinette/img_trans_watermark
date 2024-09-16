#! /bin/bash

MODEL="pan"
BACKBONE="mobilenet_v2"
DATASET="optic"
DATA_PATH="RIM-ONE_DL"
EXP="rq5_flower"

echo "RQ5: ${MODEL} ${DATASET} Semseg"
python src/train_pylight_semseg.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/train/${DATASET}" \
    --metrics_fname "${EXP}_${MODEL}_semseg_${DATASET}_train" \
    --epochs 200 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}