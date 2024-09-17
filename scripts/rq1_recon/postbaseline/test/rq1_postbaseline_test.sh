#! /bin/bash

DATASET="cifar"
DATA_PATH=None

echo "Post-Baseline ${DATASET} Test........................................"
# Trigger color - Purple
python src/train_spatial_post_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --save_path "drive/MyDrive/SatML_watermarking/rq1/postbaseline/test/${DATASET}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}" \
    --Hnet "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}/Hnet.pth" \
    --Rnet "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}/Rnet.pth" \
    --Dnet "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}/Dnet.pth" \
    --metrics_fname "rq1_postbaseline_${DATASET}_test" \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH}


DATASET="clwd"
DATA_PATH="CLWD"

echo "Post-Baseline ${DATASET} Test........................................"
# Trigger color - Purple
python src/train_spatial_post_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --save_path "drive/MyDrive/SatML_watermarking/rq1/postbaseline/test/${DATASET}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}" \
    --Hnet "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}/Hnet.pth" \
    --Rnet "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}/Rnet.pth" \
    --Dnet "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}/Dnet.pth" \
    --metrics_fname "rq1_postbaseline_${DATASET}_test" \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH}

DATASET="imagenet"
DATA_PATH="ImageNet"

echo "Post-Baseline ${DATASET} Test........................................"
# Trigger color - Purple
python src/train_spatial_post_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --save_path "drive/MyDrive/SatML_watermarking/rq1/postbaseline/test/${DATASET}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}" \
    --Hnet "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}/Hnet.pth" \
    --Rnet "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}/Rnet.pth" \
    --Dnet "drive/MyDrive/SatML_watermarking/rq1/postbaseline/train/${DATASET}/Dnet.pth" \
    --metrics_fname "rq1_postbaseline_${DATASET}_test" \
    --test \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH}