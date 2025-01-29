#!/usr/bin/env bash

# test.sh
# A simple script to run a minimal CPU-based test of MorphoDiff's train.py

# Activate your environment if you want in-script
# (optional if you’re already in the conda env interactively)
# module load Mambaforge/23.3.1-1-hpc1-bdist
# conda activate /proj/aicell/users/x_aleho/conda_envs/morphodiff

accelerate launch \
  --mixed_precision="no" \
  morphodiff/train.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-4" \
  --naive_conditional="conditional" \
  --train_data_dir="/proj/aicell/users/x_aleho/MorphoDiff/datasets/BBBC021/experiment_01_resized/train_imgs" \
  --dataset_id="BBBC021_experiment_01_resized" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --image_column="image" \
  --caption_column="additional_feature" \
  --validation_prompts="aphidicolin" \
  --pretrained_vae_path="stable-diffusion-v1-4" \
  --cache_dir="/tmp/" \
  --seed=42 \
  --report_to="wandb" \
  --use_ema \
  --output_dir="/proj/aicell/users/x_aleho/MorphoDiff/training_out" \
  --checkpointing_log_file="/proj/aicell/users/x_aleho/MorphoDiff/training_out/logs.csv"