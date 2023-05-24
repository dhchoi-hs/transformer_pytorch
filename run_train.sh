#!/usr/bin/env bash


nohup python train_lm_encoder.py -c=configs/config_ln_encoder.yaml -d=output/v5k_lr_e04 -m="dataset 1/5, lr scheduler" > /dev/null &

