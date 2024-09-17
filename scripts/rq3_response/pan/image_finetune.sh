MODEL="pan"
BACKBONE="mobilenet_v2"
DATASET="imagenet"
DATA_PATH="ImageNet"
EXP="rq3"
RESPONSE="patch_image"

echo "${EXP}: ${MODEL} ${DATASET} (${RESPONSE}) finetune ......................................."
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --image_signature flower \
    --save_path "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/train/${DATASET}" \
    --checkpoint "drive/MyDrive/SatML_watermarking/${EXP}/${MODEL}/${RESPONSE}/train/${DATASET}/${MODEL}_pu-tl-small_gr-tr-small" \
    --metrics_fname "${EXP}_${MODEL}_${RESPONSE}_${DATASET}_train" \
    --epochs 10 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}