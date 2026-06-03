#!/bin/bash
#SBATCH -A <your_slurm_account>
#SBATCH -p <your_partition>
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --job-name=llava_baseline
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================
# Vanilla fp16/bf16 LLaVA-1.5-7B finetune on ShareGPT4V cap100k.
# NO QAT — used to verify data + env are healthy. If this gives
# a normal loss curve (~2-3 at step 10, dropping smoothly), then
# QAT is the broken piece, not the data.
# ============================================================

set -euo pipefail

cd "${HOME}"
module purge
module load profile/deeplrn
module load cuda/12.3
module load gcc/12.2.0
export PYTHONPATH=""
source "${HOME}/llava/bin/activate"

PROJECT_DIR=${PROJECT_DIR:-"${HOME}/GRACE"}
DATA_ROOT=${DATA_ROOT:-"/path/to/ShareGPT4V"}
DATA_PATH=${DATA_PATH:-"${DATA_ROOT}/sharegpt4v_instruct_gpt4-vision_cap100k.json"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"${DATA_ROOT}/data"}

SCRATCH_ROOT=${SCRATCH_ROOT:-"/path/to/scratch"}
HF_LOCAL_CACHE=${HF_LOCAL_CACHE:-"${SCRATCH_ROOT}/hf_cache"}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${SCRATCH_ROOT}/llava-v1.5-7b"}
VISION_TOWER=${VISION_TOWER:-"${SCRATCH_ROOT}/clip-vit-large-patch14-336"}
CONV_VERSION=${CONV_VERSION:-"v1"}
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO:-"pad"}

export HF_HOME="${HF_LOCAL_CACHE}"
export TRANSFORMERS_CACHE="${HF_LOCAL_CACHE}"
export HF_DATASETS_CACHE="${HF_LOCAL_CACHE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_MODE=offline
export WANDB_DIR="${SCRATCH_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"

LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}

CKPT_ROOT=${CKPT_ROOT:-"/path/to/scratch/checkpoints"}
OUTPUT_DIR=${OUTPUT_DIR:-"${CKPT_ROOT}/llava-baseline-bf16"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${PROJECT_DIR}/scripts/zero2.json"}

mkdir -p "${OUTPUT_DIR}"

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29501}
export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_DIR}"

echo "=============================="
echo "LLaVA BASELINE (no QAT)"
echo "=============================="
echo "MODEL=${MODEL_NAME_OR_PATH}"
echo "DATA =${DATA_PATH}"
echo "IMGS =${IMAGE_FOLDER}"
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "NODES=${SLURM_NNODES}, GPUs/node=4"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "JOB_ID=${SLURM_JOB_ID}"

srun --kill-on-bad-exit=1 \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    torchrun \
        --nproc_per_node=4 \
        --nnodes="${SLURM_NNODES}" \
        --rdzv_id="${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m llava.train.train_mem \
        --deepspeed "${DEEPSPEED_CONFIG}" \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --version "${CONV_VERSION}" \
        --data_path "${DATA_PATH}" \
        --image_folder "${IMAGE_FOLDER}" \
        --vision_tower "${VISION_TOWER}" \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio "${IMAGE_ASPECT_RATIO}" \
        --group_by_modality_length True \
        --bf16 True \
        --tf32 True \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --learning_rate "${LEARNING_RATE}" \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 5 \
        --model_max_length "${MODEL_MAX_LENGTH}" \
        --gradient_checkpointing True \
        --dataloader_num_workers 2 \
        --lazy_preprocess True \
        --ddp_find_unused_parameters False \
        --remove_unused_columns False \
        --report_to wandb

echo "Baseline training complete."
