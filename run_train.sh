#!/usr/bin/env bash


nohup python train.py \
    -c=configs/config_ln_encoder2.yaml \
    -d=/data/dhchoi/trained/sunny_side_up/pile_v20k_bat128_d1024_h8_lyr6_ff2048_lr1e-4_rlop_dropout0.1_use_silu_mlflow \
    -t=pre-train > /dev/null &


nohup python train.py \
    --fine_tuning_config=configs/config_fine_tuning.yaml \
    --train_type=fine-tuning \
    --model_dir=/home/dhchoi/projects/transformer_pytorch2/output/fine_tuning_aihub_unfreeze1_no_keyword_no_mention  > /dev/null &

python train.py \
    --pre_trained_config=/data/dhchoi/trained/sunny_side_up/pile_v10k_bat128_d1024_h8_lyr6_ff2048_lr1e-4_rlop_dropout0.1_use_silu_mlflow/config_ln_encoder.yaml \
    --fine_tuning_config=configs/config_fine_tuning.yaml \
    --train_type=fine-tuning