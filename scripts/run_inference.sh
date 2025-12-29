#!/bin/bash

# Input and output directories
INPUT_DIR=images_for_depth_inference
OUTPUT_DIR=depth_results/

# Pretrained DispNet
DISP_NET=checkpoints/dispnet112_model_best.pth.tar

# Run depth inference
python3 run_inference.py \
    --pretrained $DISP_NET \
    --resnet-layers 50 \
    --dataset-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --output-disp
