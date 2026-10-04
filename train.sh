#!/usr/bin/env bash
# DDP launch via torchrun. Resumes from epoch_2.pth on the
# hot-sapiens-16batch config (TRAIN.start_epoch=2).
#
# To start fresh, drop the TRAIN.start_epoch override.
# To change GPU set, edit CUDA_VISIBLE_DEVICES and --nproc_per_node.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    --cfg config/hot-sapiens-hyhot.yaml \
