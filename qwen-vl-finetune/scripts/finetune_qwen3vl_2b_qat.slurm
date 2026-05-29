#!/bin/bash
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=8:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --job-name=qwen3vl_qat
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================
# Qwen3-VL-2B QAT (Quantization-Aware Training) — baseline.
# Identical knobs to finetune_qwen3vl_2b_grace.slurm but WITHOUT
# the teacher / GRACE distillation losses. Useful as the ablation
# baseline for the GRACE paper.
#
# - Default data: sharegpt4v_mix665k under SHAREGPT4V_ROOT
#   (override via DATASETS env var).
# - Default: W4, group_size=128 (override via QAT_BITS / QAT_GROUP_SIZE)
# - 4 nodes x 4 A100-80GB = 16 GPUs, DeepSpeed ZeRO-2, bf16
# - Auto-resume: if OUTPUT_DIR has checkpoint-*, resume from the latest.
# ============================================================

set -euo pipefail

# -------- Environment (Leonardo / CINECA, clean venv) --------
cd "${HOME}"
module purge
module load profile/deeplrn
module load cuda/12.3
module load gcc/12.2.0
# Same python binary the venv was created with — venv only stores a path
# to its interpreter, so this dir must be reachable on every compute node.
export PATH="${HOME}/python3.11/bin:${PATH}"
export PYTHONPATH=""
source "${HOME}/qwen3vl/bin/activate"

# -------- Paths --------
PROJECT_DIR=${PROJECT_DIR:-"${HOME}/Qwen2-VL/qwen-vl-finetune"}

# Compute nodes have NO internet — everything must be on scratch already.
SCRATCH_ROOT=${SCRATCH_ROOT:-"/leonardo_scratch/large/userexternal/ychen003"}
HF_LOCAL_CACHE=${HF_LOCAL_CACHE:-"${SCRATCH_ROOT}/hf_cache"}

# -------- Model + data --------
# Student init: base Qwen3-VL-2B-Instruct.
BASE_MODEL_DIR=${BASE_MODEL_DIR:-"${SCRATCH_ROOT}/Qwen3-VL-2B-Instruct"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${BASE_MODEL_DIR}"}

# Loader reads paths from qwenvl/data/__init__.py.
# SHAREGPT4V_ROOT overrides the ShareGPT4V root at runtime.
export SHAREGPT4V_ROOT=${SHAREGPT4V_ROOT:-"${SCRATCH_ROOT}/ShareGPT4V"}
DATASETS=${DATASETS:-"sharegpt4v_mix665k"}

# -------- Force fully offline mode on compute nodes --------
export HF_HOME="${HF_LOCAL_CACHE}"
export TRANSFORMERS_CACHE="${HF_LOCAL_CACHE}"
export HF_DATASETS_CACHE="${HF_LOCAL_CACHE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# -------- W&B offline (sync later from login node) --------
export WANDB_MODE=offline
export WANDB_DIR="${SCRATCH_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"

# -------- QAT config --------
QAT_BITS=${QAT_BITS:-4}                # 4 or 8
QAT_GROUP_SIZE=${QAT_GROUP_SIZE:-128}
QAT_SCALE_LR=${QAT_SCALE_LR:-2e-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
# 4 nodes × 4 GPU × per_device=4 × grad_accum=4  ⇒  effective batch ~256
# 665k / 256 ≈ 2600 steps/epoch, so SAVE_STEPS=500 → ~5 saves per epoch.
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}

CKPT_ROOT=${CKPT_ROOT:-"${SCRATCH_ROOT}/checkpoints"}
RUN_NAME=${RUN_NAME:-"qwen3vl-2b-qat-w${QAT_BITS}g${QAT_GROUP_SIZE}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${CKPT_ROOT}/${RUN_NAME}"}
# ZeRO-2 (not ZeRO-3): 2B fits trivially on A100-80GB, and ZeRO-3 partitions
# log_scale across ranks BEFORE our eager init can hit the sharded storage,
# which leaves scales at 0 → round(W/1) crushes weights → loss starts ~60.
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${PROJECT_DIR}/scripts/zero2.json"}

mkdir -p "${OUTPUT_DIR}"

# -------- Multi-node rendezvous --------
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL / PyTorch tuning for multi-node long-seq VLM.
# PyTorch 2.6 renamed these env vars (TORCH_* prefix is now authoritative).
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1            # abort on collective error
export TORCH_NCCL_BLOCKING_WAIT=1                   # wait() blocks; watchdog fires
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900         # 15 min → fail faster than 30 min default
export NCCL_IB_TIMEOUT=22
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_DIR}"

echo "=============================="
echo "Qwen3-VL-2B QAT  (W${QAT_BITS}, group=${QAT_GROUP_SIZE})"
echo "=============================="
echo "MODEL=${MODEL_NAME_OR_PATH}"
echo "DATASET=${DATASETS}  (root=${SHAREGPT4V_ROOT})"
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "NODES=${SLURM_NNODES}, GPUs/node=4, Total=$((SLURM_NNODES * 4))"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "JOB_ID=${SLURM_JOB_ID}"

# When OUTPUT_DIR already has checkpoints, load the model from the latest
# checkpoint. HuggingFace Trainer's `from_pretrained` pass still runs even
# when resuming, so MODEL_NAME_OR_PATH must point at a dir that actually
# contains model.safetensors.
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [ -n "${LATEST_CKPT}" ] && [ -d "${LATEST_CKPT}" ]; then
    echo "RESUME: Loading model from latest checkpoint: ${LATEST_CKPT}"
    MODEL_NAME_OR_PATH="${LATEST_CKPT}"

    # Backfill preprocessor configs if missing (older checkpoints predate
    # SaveProcessorCallback). AutoProcessor.from_pretrained() needs these.
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

srun --kill-on-bad-exit=1 \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    torchrun \
        --nproc_per_node=4 \
        --nnodes="${SLURM_NNODES}" \
        --rdzv_id="${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m qwenvl.train.train_qwen_qat \
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
        --max_pixels 50176 \
        --min_pixels 784 \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --learning_rate "${LEARNING_RATE}" \
        --weight_decay 1e-3 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1 \
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
        --report_to wandb \
        --qat_enable True \
        --qat_bits "${QAT_BITS}" \
        --qat_group_size "${QAT_GROUP_SIZE}" \
        --qat_scale_lr "${QAT_SCALE_LR}" \
        --qat_save_quantized True

echo "Qwen3-VL-2B QAT complete (output at ${OUTPUT_DIR})."
