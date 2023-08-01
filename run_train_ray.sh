#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0
export TUNE_DISABLE_STRICT_METRIC_CHECKING=1

nohup python train_lm_encoder_ray_tune.py -c=configs/config_ln_encoder.yaml -d=output/raytunetunetune > /dev/null & 
