#! /bin/bash

DATASET="imagenet"
DATA_PATH="ImageNet"
METHOD="prebaseline"
ATTACK="ftune1"

echo "RQ1 ${METHOD} ${DATASET} ATTACK: ${ATTACK} ........................................"
# Trigger color - Purple
python src/train_spatial_pre_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}" \
    --Hnet "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}/Hnet.pth" \
    --Rnet "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}/Rnet.pth" \
    --Dnet "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}/Dnet.pth" \
    --metrics_fname "rq1_${METHOD}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH}
    

ATTACK="ftune5"

echo "RQ1 ${METHOD} ${DATASET} ATTACK: ${ATTACK} ........................................"
# Trigger color - Purple
python src/train_spatial_pre_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}" \
    --Hnet "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}/Hnet.pth" \
    --Rnet "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}/Rnet.pth" \
    --Dnet "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}/Dnet.pth" \
    --metrics_fname "rq1_${METHOD}_${DATASET}_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH}

ATTACK="overwrite"

echo "RQ1 ${METHOD} ${DATASET} ATTACK: ${ATTACK} ........................................"
# Trigger color - Purple
python src/train_spatial_pre_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/attack/${DATASET}_${ATTACK}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}" \
    --Hnet "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}/Hnet.pth" \
    --Rnet "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}/Rnet.pth" \
    --Dnet "drive/MyDrive/SatML_watermarking/rq1/${METHOD}/train/${DATASET}/Dnet.pth" \
    --metrics_fname "rq1_${METHOD}_${DATASET}_attack_${ATTACK}" \
    --image_signature flower \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH}