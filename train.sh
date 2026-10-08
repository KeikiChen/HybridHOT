#!/usr/bin/env bash
# DDP training with torchrun on 8 GPUs (edit CUDA_VISIBLE_DEVICES / --nproc_per_node to change).
# Usage (from HybridHOT_release/): sh ./train.sh

CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --standalone \
    --nproc_per_node=1 \
    train.py \
    --cfg config/hot-sapiens-hyhot.yaml \
