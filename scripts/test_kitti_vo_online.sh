#!/bin/bash

# Dataset and output paths
DATASET_DIR=/scratch/data/kitti/data_odometry_color/dataset/sequences/
OUTPUT_DIR=vo_results_online/

# Pretrained models
POSE_NET=checkpoints/exp_pose112_model_best.pth.tar
DISP_NET=checkpoints/dispnet112_model_best.pth.tar

# Run online VO test
CUDA_VISIBLE_DEVICES=0 \
python test_vo_online.py \
    -p 1 \
    -s 0.1 \
    -c 0.5 \
    --img-height 256 \
    --img-width 832 \
    --sequence 09 \
    --pretrained-posenet $POSE_NET \
    --pretrained-disp $DISP_NET \
    --dataset-dir $DATASET_DIR \
    --output-dir $OUTPUT_DIR \
    --epochs 2 \
    --lr 1e-4 \
    --sequence-length 3 \
    --resnet-layers 50 \
    --select-best 1 \
    --with-mask 1 \
    --with-auto-mask 1 \
    --padding-mode 'border'  \
    # --thread       # Optional multi-thread pre-loading
    # --part         # Optional partial online update
    # --output-disp  # Optional output disparity

# Evaluate VO results
python ./kitti_eval/eval_odom.py --result=$OUTPUT_DIR --align='7dof'
