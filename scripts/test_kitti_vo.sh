#!/bin/bash

# Dataset and output paths
DATASET_DIR=/home/areeba/ICRAMaxxing/SG-VO/data/kitti_odom/sequences/
OUTPUT_DIR=vo_results/

# Pretrained pose network
POSE_NET=/home/areeba/ICRAMaxxing/SG-VO/checkpoints/exp_pose112_model_best.pth.tar

# Run VO test for selected sequences
for sequence in $(seq -w 00 10); do
    CUDA_VISIBLE_DEVICES=0 \
    python test_vo.py \
        --img-height 256 \
        --img-width 832 \
        --sequence $sequence \
        --pretrained-posenet $POSE_NET \
        --dataset-dir $DATASET_DIR \
        --output-dir $OUTPUT_DIR \
        # --thread   # Optional multi-thread pre-loading
done

# Evaluate VO results
python ./kitti_eval/eval_odom.py --result=$OUTPUT_DIR --align='7dof'
