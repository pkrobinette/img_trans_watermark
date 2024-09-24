#! /bin/bash

MODEL="linknet"
BACKBONE="mobilenet_v2"
DATASET="cifar"
DATASET_PATH=None

echo "DEMO: ${MODEL} CIFAR-10 (TRAIN) ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path 'demo_results/train' \
    --metrics_fname 'demo_train' \
    --epochs 3 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATASET_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}
    

ATTACK="ftune1"
echo "DEMO ATTACK: Finetune 1 epoch ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "demo_results/attack/${ATTACK}" \
    --checkpoint "demo_results/train/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "demo_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATASET_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ATTACK="ftune5"
echo "DEMO ATTACK: Finetune 5 epochs ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "demo_results/attack/${ATTACK}" \
    --checkpoint "demo_results/train/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "demo_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATASET_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

ATTACK="overwrite"
echo "DEMO ATTACK: Overwrite  ........................................"
# Trigger color - Purple
python src/train_pylight.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --response_c green \
    --response_s full \
    --save_path "demo_results/attack/${ATTACK}" \
    --checkpoint "demo_results/train/${MODEL}_pu-tl-small_gr-tr-full" \
    --metrics_fname "demo_attack_${ATTACK}" \
    --attack ${ATTACK} \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset ${DATASET} \
    --data_path ${DATASET_PATH} \
    --model_type ${MODEL} \
    --backbone ${BACKBONE}

echo -e "\n\nMaking table .............................................\n\n"
python make_demo_table.py


echo -e "\n\n %%%%%%%%%%%%% All results saved to demo_results folder %%%%%%%%%%%%%%%\n\n"
