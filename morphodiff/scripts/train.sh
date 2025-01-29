#!/bin/bash
#SBATCH -A berzelius-2024-230
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train_morphodiff
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# handler()
# {
# echo "function handler called at $(date)"
# scontrol requeue $SLURM_JOB_ID
# }
# trap 'handler' SIGUSR1

# activate the environment
# source /home/env/morphodiff/bin/activate
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/morphodiff


## Fixed parameters ##
export CKPT_NUMBER=0
export TRAINED_STEPS=0

# set "conditional" for training MorphoDiff, and "naive" for training Stable Diffuison
export SD_TYPE="conditional"

# set the path to the pretrained VAE model. Downloaded from: https://huggingface.co/CompVis/stable-diffusion-v1-4 
export VAE_DIR="/stable-diffusion-v1-4"

# set the path to the log directory
export LOG_DIR="model/log/"
# chek if LOG_DIR exists, if not create it
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p $LOG_DIR
fi

# set the experiment name
export EXPERIMENT="BBBC021_experiment_01_resized"

# set the path to the pretrained model, which could be either pretrained Stable Diffusion, or a pretrained MorphoDiff model
export MODEL_NAME="/stable-diffusion-v1-4"

# set the path to the training data directory. Folder contents must follow the structure described in"
# " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
# " must exist to provide the captions for the images. Ignored if `dataset_name` is specified.
# export TRAIN_DIR="/datasets/BBBC021/experiment_01_resized/train_imgs/"
export TRAIN_DIR="/proj/aicell/users/x_aleho/MorphoDiff/datasets/BBBC021/experiment_01_resized/train_imgs"

# set the path to the checkpointing log file in .csv format. Should change the MorphoDiff to SD if training unconditional Stable Diffusion 
export CKPT_LOG_FILE="${LOG_DIR}${EXPERIMENT}_log/${EXPERIMENT}_MorphoDiff_checkpoints.csv"

# set the validation prompts/perturbation ids, separated by ,
# export VALID_PROMPT="cytochalasin-d,docetaxel,epothilone-b"
export VALID_PROMPT="aphidicolin,colchicine,cytochalasin-b,doxorubicin"

# the header for the checkpointing log file
export HEADER="dataset_id,log_dir,pretrained_model_dir,checkpoint_dir,seed,trained_steps,checkpoint_number"
mkdir -p ${LOG_DIR}${EXPERIMENT}_log

# Function to get column index by header name
get_column_index() {
    local header_line=$1
    local column_name=$2
    echo $(echo "$header_line" | tr ',' '\n' | nl -v 0 | grep "$column_name" | awk '{print $1}')
}

# Check if the checkpointing log CSV file exists
if [ ! -f "$CKPT_LOG_FILE" ]; then
    # If the file does not exist, create it and add the header
    echo "$HEADER" > "$CKPT_LOG_FILE"
    echo "CSV checkpointing log file created with header: $HEADER"

elif [ $(wc -l < "$CKPT_LOG_FILE") -eq 1 ]; then
    # overwrite the header line
    echo "$HEADER" > "$CKPT_LOG_FILE"
    echo "CSV checkpointing log file header overwritten with: $HEADER"

else
    echo "CSV checkpointing log file exists in $CKPT_LOG_FILE"
    echo "Reading the last line of the log file to resume training"
    # If the file exists, read the last line
    LAST_LINE=$(tail -n 1 "$CKPT_LOG_FILE")
    
    # Extract the header line to determine the index of "checkpoint_dir" column
    HEADER_LINE=$(head -n 1 "$CKPT_LOG_FILE")
    CHECKPOINT_DIR_INDEX=$(get_column_index "$HEADER_LINE" "checkpoint_dir")

    # Extract the checkpoint_dir value from the last line
    MODEL_NAME=$(echo "$LAST_LINE" | cut -d',' -f$(($CHECKPOINT_DIR_INDEX + 1)))

    # Extract the last column from the last line
    LAST_COLUMN=$(echo "$LAST_LINE" | awk -F',' '{print $NF}')
    # Convert the last column to an integer
    CKPT_NUMBER=$((LAST_COLUMN))

    # get the number of trained steps so far
    TRAINED_STEPS_INDEX=$(get_column_index "$HEADER_LINE" "trained_steps")
    TRAINED_STEPS=$(echo "$LAST_LINE" | cut -d',' -f$(($TRAINED_STEPS_INDEX + 1)))

fi

# add 1 to the value of CKPT_NUMBER
export CKPT_NUMBER=$((${CKPT_NUMBER}+1))
export OUTPUT_DIR="/checkpoint/${USER}/${SLURM_JOB_ID}/${EXPERIMENT}-MorphoDiff"

echo "Checkpoint number: $CKPT_NUMBER"
echo "Model directory: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Data directory: $TRAIN_DIR"
echo "Trained steps: $TRAINED_STEPS"


accelerate launch --mixed_precision="fp16" ../train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --naive_conditional=$SD_TYPE \
  --train_data_dir=$TRAIN_DIR \
  --dataset_id=$EXPERIMENT \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 \
  --random_flip \
  --use_ema \
  --train_batch_size=32 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=500 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_epochs=20 \
  --validation_prompts=$VALID_PROMPT  \
  --checkpointing_steps=20 \
  --output_dir=$OUTPUT_DIR \
  --image_column="image" \
  --caption_column='additional_feature' \
  --pretrained_vae_path=$VAE_DIR \
  --cache_dir="/tmp/" \
  --report_to="wandb" \
  --logging_dir="${LOG_DIR}${EXPERIMENT}_log" \
  --seed=42 \
  --checkpointing_log_file=$CKPT_LOG_FILE \
  --checkpoint_number=$CKPT_NUMBER \
  --trained_steps=$TRAINED_STEPS


# Requeue the job
# echo `date`: Job $SLURM_JOB_ID finished running
# scontrol requeue $SLURM_JOB_ID
# echo `date`: Job $SLURM_JOB_ID reallocated resource