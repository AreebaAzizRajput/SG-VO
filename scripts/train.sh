#!/bin/bash

# Dataset path
TRAIN_SET=/home/areeba/ICRAMaxxing/SG-VO/data/kitti_256

# GPU
CUDA_VISIBLE_DEVICES=0

# Run training
python train.py $TRAIN_SET \
    --resnet-layers 50 \
    --num-scales 1 \
    -b 4 \
    -p 1 \
    -s 0.1 \
    -c 0.5 \
    --epochs 200 \
    --epoch-size 1000 \
    --sequence-length 3 \
    --with-ssim 1 \
    --with-mask 1 \
    --with-auto-mask 1 \
    --with-pretrain 1 \
    --log-output \
    --name resnet50 \
    --workers 4 \
    --padding-mode 'border' \
    --lr 1e-4
