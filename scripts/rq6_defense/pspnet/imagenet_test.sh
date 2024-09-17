#! /bin/bash

DATASET="imagenet"
DATA_PATH="ImageNet"
EXP="rq6_defense"
MODEL="pspnet"
BACKBONE="mobilenet_v2"

TRIGGER_C="purple"
TRIGGER_POS="top_right"

echo "RQ1 ImageNet TEST: ${TRIGGER_C}-${TRIGGER_POS} ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c ${TRIGGER_C} \
    --trigger_pos ${TRIGGER_POS} \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER_C}-${TRIGGER_POS}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "${EXP}_${DATASET}_${TRIGGER_C}_${TRIGGER_POS}" \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

TRIGGER_C="purple"
TRIGGER_POS="bottom_left"

echo "RQ1 ImageNet TEST: ${TRIGGER_C}-${TRIGGER_POS} ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c ${TRIGGER_C} \
    --trigger_pos ${TRIGGER_POS} \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER_C}-${TRIGGER_POS}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "${EXP}_${DATASET}_${TRIGGER_C}_${TRIGGER_POS}" \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

TRIGGER_C="purple"
TRIGGER_POS="bottom_right"

echo "RQ1 ImageNet TEST: ${TRIGGER_C}-${TRIGGER_POS} ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c ${TRIGGER_C} \
    --trigger_pos ${TRIGGER_POS} \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER_C}-${TRIGGER_POS}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "${EXP}_${DATASET}_${TRIGGER_C}_${TRIGGER_POS}" \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

TRIGGER_C="purple"
TRIGGER_POS="center"

echo "RQ1 ImageNet TEST: ${TRIGGER_C}-${TRIGGER_POS} ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c ${TRIGGER_C} \
    --trigger_pos ${TRIGGER_POS} \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER_C}-${TRIGGER_POS}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "${EXP}_${DATASET}_${TRIGGER_C}_${TRIGGER_POS}" \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

TRIGGER_C="blue"
TRIGGER_POS="top_left"

echo "RQ1 ImageNet TEST: ${TRIGGER_C}-${TRIGGER_POS} ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c ${TRIGGER_C} \
    --trigger_pos ${TRIGGER_POS} \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER_C}-${TRIGGER_POS}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "${EXP}_${DATASET}_${TRIGGER_C}_${TRIGGER_POS}" \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

TRIGGER_C="purple"
TRIGGER_POS="top_left"
SIZE="full"

echo "RQ1 ImageNet TEST: ${TRIGGER_C}-${TRIGGER_POS} ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c ${TRIGGER_C} \
    --trigger_pos ${TRIGGER_POS} \
    --trigger_s  ${SIZE} \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${TRIGGER_C}-${TRIGGER_POS}-${SIZE}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${MODEL}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "${EXP}_${DATASET}_${TRIGGER_C}_${TRIGGER_POS}_${SIZE}" \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}