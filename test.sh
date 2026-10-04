#!/usr/bin/env bash

EPOCH_NUM=20
for epoch in `seq 1 $EPOCH_NUM`
do
    CUDA_VISIBLE_DEVICES=0 python eval_metric_epoch.py --cfg config/hot-sapiens-hyhot.yaml --epoch ${epoch}
done