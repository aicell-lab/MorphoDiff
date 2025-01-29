from local to cluster:

```
rsync -av --progress \
 --exclude 'logs/' \
 --exclude '\*.pyc' \
 --exclude '**pycache**' \
 --exclude '.git' \
 --exclude 'wandb' \
 --exclude 'env' \
 --exclude 'data/BBBC021/images' \
 /Users/lapuerta/aicell/MorphoDiff x_aleho@berzelius1.nsc.liu.se:/proj/aicell/users/x_aleho/MorphoDiff
```

running train.py directly:

```
accelerate launch \
  --mixed_precision="no" \
  train.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-4" \
  --naive_conditional="conditional" \
  --train_data_dir="/proj/aicell/users/x_aleho/MorphoDiff/datasets/BBBC021/experiment_01_resized/train_imgs" \
  --dataset_id="BBBC021_experiment_01_resized" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=5 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --image_column="image" \
  --caption_column="additional_feature" \
  --pretrained_vae_path="stable-diffusion-v1-4" \
  --cache_dir="/tmp/" \
  --seed=42 \
  --report_to="none" \
  --use_ema
```

load mamba:
`module load Mambaforge/23.3.1-1-hpc1-bdist`
