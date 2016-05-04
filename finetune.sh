#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
train_dir=./finetune_loss_weight1
pretrained_dir=./train_loss_weight1

python train.py --finetune True \
    --train_dir $train_dir \
    --pretrained_dir $pretrained_dir \
    --batch_size 64 \
    --max_steps 25000 \
    --initial_lr 0.01 \
    --lr_step_epoch 75.0 \
    --lr_decay 0.1 \
    --l2_weight 0.0005 \
    --l1_weight 0.0 \
    --momentum 0.9 \
    --gpu_fraction 0.96 \
    --basenet_train True \
    --basenet_lr_ratio 0.01 \
    --checkpoint_interval 1000 \
