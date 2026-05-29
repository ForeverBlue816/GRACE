#!/bin/bash
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --job-name=qwen3vl_bf16
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================
# Qwen3-VL-2B BF16 fine-tuning (NO quantization).
#
# - Starting point: the base Qwen3-VL-2B-Instruct HF model dir.
# - Data: ShareGPT4V mix665k (override via DATASETS env var; comma-separate
#   to combine with sharegpt4v_gpt4v_cap100k).
# - Goal: produce a BF16 student fine-tune to be used as a baseline
#   or as initialization for downstream QAT / distillation runs.
# - 4 nodes x 4 A100-80GB = 16 GPUs, DeepSpeed ZeRO-2, bf16.
# - Auto-resume: if OUTPUT_DIR already contains checkpoint-*, load the
#   latest checkpoint as the model source AND resume trainer state.
# ============================================================

set -euo pipefail

# -------- Environment (Leonardo / CINECA, clean venv) --------
cd "${HOME}"
module purge
module load profile/deeplrn
module load cuda/12.3
module load gcc/12.2.0
export PATH="${HOME}/python3.11/bin:${PATH}"
export PYTHONPATH=""
source "${HOME}/qwen3vl/bin/activate"

# -------- Paths --------
PROJECT_DIR=${PROJECT_DIR:-"${HOME}/Qwen2-VL/qwen-vl-finetune"}
SCRATCH_ROOT=${SCRATCH_ROOT:-"/leonardo_scratch/large/userexternal/ychen003"}
HF_LOCAL_CACHE=${HF_LOCAL_CACHE:-"${SCRATCH_ROOT}/hf_cache"}
CKPT_ROOT=${CKPT_ROOT:-"${SCRATCH_ROOT}/checkpoints"}

# -------- Base model --------
BASE_MODEL_DIR=${BASE_MODEL_DIR:-"${SCRATCH_ROOT}/Qwen3-VL-2B-Instruct"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${BASE_MODEL_DIR}"}

# -------- Data --------
# Comma-separated; appending %NN downsamples that dataset to NN%.
# Example: DATASETS="sharegpt4v_mix665k,sharegpt4v_gpt4v_cap100k%50"
DATASETS=${DATASETS:-"sharegpt4v_mix665k"}
export SHAREGPT4V_ROOT=${SHAREGPT4V_ROOT:-"${SCRATCH_ROOT}/ShareGPT4V"}

# -------- Offline mode (compute nodes have no internet) --------
export HF_HOME="${HF_LOCAL_CACHE}"
export TRANSFORMERS_CACHE="${HF_LOCAL_CACHE}"
export HF_DATASETS_CACHE="${HF_LOCAL_CACHE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# -------- W&B offline --------
export WANDB_MODE=offline
export WANDB_DIR="${SCRATCH_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"

# -------- Training hyperparameters --------
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-3}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
# 16 GPUs * batch 4 * accum 4  = effective batch 256
# ~665k samples / 256 ≈ 2600 steps/epoch.
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}

RUN_NAME=${RUN_NAME:-"qwen3vl-2b-bf16-mixed"}
OUTPUT_DIR=${OUTPUT_DIR:-"${CKPT_ROOT}/${RUN_NAME}"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${PROJECT_DIR}/scripts/zero2.json"}

mkdir -p "${OUTPUT_DIR}"

# -------- Multi-node rendezvous --------
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29503}     # distinct from QAT stage-1/2/3 ports
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL / PyTorch tuning for multi-node long-seq VLM.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
export NCCL_IB_TIMEOUT=22
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_DIR}"

# -------- Resume handling: prefer latest checkpoint in OUTPUT_DIR --------
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [ -n "${LATEST_CKPT}" ] && [ -d "${LATEST_CKPT}" ]; then
    echo "RESUME: Loading model from latest checkpoint: ${LATEST_CKPT}"
    MODEL_NAME_OR_PATH="${LATEST_CKPT}"

    # Backfill preprocessor configs if a resumed checkpoint predates them.
    for f in preprocessor_config.json video_preprocessor_config.json; do
        if [ ! -f "${LATEST_CKPT}/${f}" ] && [ -f "${BASE_MODEL_DIR}/${f}" ]; then
            cp -v "${BASE_MODEL_DIR}/${f}" "${LATEST_CKPT}/${f}"
        fi
    done
else
    echo "FRESH: No prior checkpoint in OUTPUT_DIR; starting from ${MODEL_NAME_OR_PATH}."
fi

if [ ! -e "${MODEL_NAME_OR_PATH}" ]; then
    echo "ERROR: MODEL_NAME_OR_PATH does not exist: ${MODEL_NAME_OR_PATH}" >&2
    exit 1
fi

echo "=============================="
echo "Qwen3-VL-2B BF16 fine-tune"
echo "=============================="
echo "MODEL=${MODEL_NAME_OR_PATH}"
echo "DATASETS=${DATASETS}  (root=${SHAREGPT4V_ROOT})"
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "NODES=${SLURM_NNODES}, GPUs/node=4, Total=$((SLURM_NNODES * 4))"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "LR=${LEARNING_RATE}  wd=${WEIGHT_DECAY}"
echo "warmup_ratio=${WARMUP_RATIO}  max_grad_norm=${MAX_GRAD_NORM}"

srun --kill-on-bad-exit=1 \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    torchrun \
        --nproc_per_node=4 \
        --nnodes="${SLURM_NNODES}" \
        --rdzv_id="${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m qwenvl.train.train_qwen \
        --deepspeed "${DEEPSPEED_CONFIG}" \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --dataset_use "${DATASETS}" \
        --data_flatten True \
        --tune_mm_vision False \
        --tune_mm_mlp True \
        --tune_mm_llm True \
        --bf16 True \
        --tf32 True \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --per_device_eval_batch_size $((PER_DEVICE_BATCH*2)) \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --max_pixels 200704 \
        --min_pixels 784 \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --learning_rate "${LEARNING_RATE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --max_grad_norm "${MAX_GRAD_NORM}" \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --model_max_length "${MODEL_MAX_LENGTH}" \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --ddp_find_unused_parameters False \
        --group_by_modality_length True \
        --dataloader_drop_last True \
        --remove_unused_columns False \
        --run_name "${RUN_NAME}" \
        --report_to wandb

echo "Qwen3-VL BF16 fine-tune complete (output at ${OUTPUT_DIR})."
