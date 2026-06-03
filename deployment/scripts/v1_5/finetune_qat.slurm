#!/bin/bash
#SBATCH -A <your_slurm_account>
#SBATCH -p <your_partition>
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --job-name=llava_qat
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================
# LLaVA QAT (group-wise LSQ, weight-only) on the LLM backbone
# - Default: W4, group_size=128 (override via env vars below)
# - Works for LLaVA 1.5 (Vicuna) and 1.6 (Vicuna / Mistral)
# - 4 nodes x 4 A100-80GB = 16 GPUs, DeepSpeed ZeRO-1 by default, bf16
# - Auto-resume from checkpoints
# ============================================================

set -euo pipefail

# -------- Environment (example module block — adapt to your cluster) --------
module purge
module load cuda/12.3
module load gcc/12.2.0
export PYTHONPATH=""
source "${LLAVA_VENV:-${HOME}/llava}/bin/activate"

# -------- Paths (override via env / `sbatch --export`) --------
PROJECT_DIR=${PROJECT_DIR:-"$(pwd)"}
DATA_ROOT=${DATA_ROOT:-"/path/to/ShareGPT4V"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"${DATA_ROOT}/data"}

# Training data: the two ShareGPT4V annotation JSONs (mix665k + cap100k).
# JSONs live at ${DATA_ROOT}; images live under ${DATA_ROOT}/data.
# LazySupervisedDataset splits DATA_PATH on commas and concatenates in order.
SHAREGPT4V_MIX_JSON=${SHAREGPT4V_MIX_JSON:-"${DATA_ROOT}/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"}
SHAREGPT4V_CAP_JSON=${SHAREGPT4V_CAP_JSON:-"${DATA_ROOT}/sharegpt4v_instruct_gpt4-vision_cap100k.json"}
DATA_PATH=${DATA_PATH:-"${SHAREGPT4V_MIX_JSON},${SHAREGPT4V_CAP_JSON}"}

# -------- Model selection (override on sbatch via env) --------
# Pre-download the base weights to a local dir if your compute nodes are offline.
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"liuhaotian/llava-v1.5-7b"}
VISION_TOWER=${VISION_TOWER:-"openai/clip-vit-large-patch14-336"}
CONV_VERSION=${CONV_VERSION:-"v1"}                    # v1 = vicuna_v1; mistral_instruct for v1.6-mistral
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO:-"pad"}       # 1.6 typically "anyres"

# -------- W&B (set WANDB_MODE=offline on air-gapped nodes) --------
export WANDB_DIR=${WANDB_DIR:-"${PROJECT_DIR}/wandb"}
mkdir -p "${WANDB_DIR}"

# -------- QAT config --------
QAT_BITS=${QAT_BITS:-4}                # 4 or 8
QAT_GROUP_SIZE=${QAT_GROUP_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-2e-5}   # weight LR
QAT_SCALE_LR=${QAT_SCALE_LR:-2e-6}     # conservative LR for sensitive log-scale params
QAT_LEARN_SCALES=${QAT_LEARN_SCALES:-True}
QAT_DEBUG_GRAD_CHECK_STEPS=${QAT_DEBUG_GRAD_CHECK_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}             # set e.g. 30 for a fresh smoke test
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
SAVE_STEPS=${SAVE_STEPS:-500}        # periodic checkpoints during full runs
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}

CKPT_ROOT=${CKPT_ROOT:-"./checkpoints"}
if [[ "${QAT_LEARN_SCALES,,}" == "true" ]]; then
    QAT_SCALE_MODE="learned-scales"
else
    QAT_SCALE_MODE="fixed-scales"
fi
OUTPUT_DIR=${OUTPUT_DIR:-"${CKPT_ROOT}/llava-qat-w${QAT_BITS}-g${QAT_GROUP_SIZE}-${QAT_SCALE_MODE}"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${PROJECT_DIR}/scripts/zero1.json"}

mkdir -p "${OUTPUT_DIR}"

# -------- Multi-node rendezvous --------
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL / PyTorch tuning for multi-node long-seq VLM
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TIMEOUT_S=7200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_DIR}"

echo "=============================="
echo "LLaVA QAT  (W${QAT_BITS}, group=${QAT_GROUP_SIZE})"
echo "=============================="
echo "MODEL=${MODEL_NAME_OR_PATH}"
echo "DATA  =${DATA_PATH}"
echo "IMGS  =${IMAGE_FOLDER}"
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "NODES=${SLURM_NNODES}, GPUs/node=4, Total=$((SLURM_NNODES * 4))"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "LR weights=${LEARNING_RATE}, log_scale=${QAT_SCALE_LR}"
echo "QAT_LEARN_SCALES=${QAT_LEARN_SCALES}"
echo "QAT_DEBUG_GRAD_CHECK_STEPS=${QAT_DEBUG_GRAD_CHECK_STEPS}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "JOB_ID=${SLURM_JOB_ID}"

if ls "${OUTPUT_DIR}"/checkpoint-* 1>/dev/null 2>&1; then
    echo "RESUME: Found checkpoints in ${OUTPUT_DIR}, will auto-resume."
else
    echo "FRESH: No checkpoints found, starting from scratch."
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
        -m llava.train.train_qat \
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
        --max_steps "${MAX_STEPS}" \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --learning_rate "${LEARNING_RATE}" \
        --max_grad_norm 1.0 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --model_max_length "${MODEL_MAX_LENGTH}" \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --ddp_find_unused_parameters False \
        --remove_unused_columns False \
        --logging_nan_inf_filter False \
        --report_to wandb \
        --qat_enable True \
        --qat_bits "${QAT_BITS}" \
        --qat_group_size "${QAT_GROUP_SIZE}" \
        --qat_scale_lr "${QAT_SCALE_LR}" \
        --qat_learn_scales "${QAT_LEARN_SCALES}" \
        --qat_debug_grad_check_steps "${QAT_DEBUG_GRAD_CHECK_STEPS}" \
        --qat_save_quantized True

echo "QAT training complete."
