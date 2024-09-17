#! /bin/bash

echo "Expr 1 Post-Baseline CLWD ATTACK: Finetune 1 epoch ........................................"
# Trigger color - Purple
python src/train_spatial_post_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/attack/postbaseline_clwd_attack_ftune1' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train' \
    --Hnet 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train/Hnet.pth' \
    --Rnet 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train/Rnet.pth' \
    --Dnet 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train/Dnet.pth' \
    --metrics_fname 'exp1_postbaseline_clwd_attack_ftune1' \
    --image_signature ECCV \
    --attack ftune1 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset clwd \
    --data_path CLWD


echo "Expr 1 Post-Baseline CLWD ATTACK: Finetune 5 epoch ........................................"
# Trigger color - Purple
python src/train_spatial_post_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/attack/postbaseline_clwd_attack_ftune5' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train' \
    --Hnet 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train/Hnet.pth' \
    --Rnet 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train/Rnet.pth' \
    --Dnet 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train/Dnet.pth' \
    --metrics_fname 'exp1_postbaseline_clwd_attack_ftune5' \
    --image_signature ECCV \
    --attack ftune5 \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset clwd \
    --data_path CLWD

echo "Expr 1 Post-Baseline CLWD ATTACK: Overwrite ........................................"
# Trigger color - Purple
python src/train_spatial_post_baseline.py \
    --trigger_c purple \
    --trigger_pos top_left \
    --trigger_s small \
    --save_path 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/attack/postbaseline_clwd_attack_overwrite' \
    --checkpoint 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train' \
    --Hnet 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train/Hnet.pth' \
    --Rnet 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train/Rnet.pth' \
    --Dnet 'drive/MyDrive/WATERMARKING/results/exp1/postbaseline/train/postbaseline_clwd_train/Dnet.pth' \
    --metrics_fname 'exp1_postbaseline_clwd_attack_overwrite' \
    --image_signature flower \
    --attack overwrite \
    --gpu \
    --num_devices 1 \
    --num_test_save 10 \
    --dataset clwd \
    --data_path CLWD