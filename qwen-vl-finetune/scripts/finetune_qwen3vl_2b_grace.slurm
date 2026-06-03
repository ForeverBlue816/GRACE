#!/bin/bash
#SBATCH -A <your_slurm_account>
#SBATCH -p <your_partition>
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --job-name=qwen3vl_grace
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================
# GRACE: Gated Relational Alignment via Confidence-based
# Distillation — QAT training for Qwen3-VL.
#
# - Teacher : Qwen3-VL-8B-Instruct  (frozen, BF16, replicated per GPU)
# - Student : Qwen3-VL-2B-Instruct  (W4/W8 group-wise LSQ QAT, trained)
# - Objective: L_total = L_CE + beta * L_GDKD + omega * L_RCKA
#     L_GDKD : confidence-gated decoupled KD (TCKD + NCKD)
#     L_RCKA : relational CKA on penultimate-layer visual tokens
#     beta   : adapted online by the IB controller
# - QAT is identical to finetune_qwen3vl_2b_qat.slurm; GRACE only adds
#   the teacher-driven loss terms.
# - 4 nodes x 4 A100-80GB = 16 GPUs, DeepSpeed ZeRO-2, bf16.
#   The 8B teacher (~16 GB BF16) sits on every GPU, NOT DeepSpeed-managed.
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
SCRATCH_ROOT=${SCRATCH_ROOT:-"/path/to/scratch"}
HF_LOCAL_CACHE=${HF_LOCAL_CACHE:-"${SCRATCH_ROOT}/hf_cache"}
CKPT_ROOT=${CKPT_ROOT:-"${SCRATCH_ROOT}/checkpoints"}

# Student init = base Qwen3-VL-2B; teacher = base Qwen3-VL-8B.
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${SCRATCH_ROOT}/Qwen3-VL-2B-Instruct"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${SCRATCH_ROOT}/Qwen3-VL-8B-Instruct"}

# -------- Data --------
# sharegpt4v_mix665k (~665k), images under <root>/data of SHAREGPT4V_ROOT.
# Comma-separate to also include sharegpt4v_gpt4v_cap100k.
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

# -------- QAT config (same knobs as finetune_qwen3vl_2b_qat) --------
QAT_BITS=${QAT_BITS:-4}                # 4 or 8
QAT_GROUP_SIZE=${QAT_GROUP_SIZE:-128}
QAT_SCALE_LR=${QAT_SCALE_LR:-2e-4}     # ~10x base LR (LSQ best practice)

# -------- GRACE config (defaults follow the paper, Table 6) --------
DKD_TEMPERATURE=${DKD_TEMPERATURE:-2.0}
DKD_ALPHA=${DKD_ALPHA:-1.0}            # TCKD weight
DKD_BETA=${DKD_BETA:-4.0}             # NCKD weight (dark knowledge)
# omega: RCKA needs real gradient share — at omega=1 the relational term
# (~0.7) is drowned 30:1 by beta*GDKD and never moves. omega=3 -> ~2.1.
RCKA_WEIGHT=${RCKA_WEIGHT:-3.0}        # omega
RCKA_LAYER=${RCKA_LAYER:--2}           # penultimate hidden_states index
# IB controller: tau must sit mid-trajectory of L_GDKD (which falls ~36 -> ~2
# over training) or the controller saturates at beta_max and stops adapting.
# beta_max is capped at 1.0 so distillation can lead but not 50x-dominate CE.
IB_TAU=${IB_TAU:-3.0}
IB_ETA=${IB_ETA:-0.003}
IB_BETA_INIT=${IB_BETA_INIT:-0.5}
IB_BETA_MIN=${IB_BETA_MIN:-0.1}
IB_BETA_MAX=${IB_BETA_MAX:-1.0}
# Linear warmup on distill terms over the first N optimizer steps.
# RCKA is noisy before the student's representation has shape; DKD is
# already adapted by the IB controller so warmup is usually unnecessary.
DKD_WARMUP_STEPS=${DKD_WARMUP_STEPS:-0}
RCKA_WARMUP_STEPS=${RCKA_WARMUP_STEPS:-400}

# -------- Training --------
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-3}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}     # paper uses 3
# Teacher (8B) is resident on every GPU, so micro-batch is kept small.
# 16 GPU x batch 2 x accum 16 = effective batch 512.
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
GRAD_ACCUM=${GRAD_ACCUM:-16}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}

RUN_NAME=${RUN_NAME:-"qwen3vl-2b-grace-w${QAT_BITS}g${QAT_GROUP_SIZE}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${CKPT_ROOT}/${RUN_NAME}"}
# ZeRO-2: 2B student fits trivially; ZeRO-3 shards log_scale before our eager
# QAT init runs (see finetune_qwen3vl_2b_qat.slurm for the full rationale).
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${PROJECT_DIR}/scripts/zero2.json"}

mkdir -p "${OUTPUT_DIR}"

# -------- Multi-node rendezvous --------
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29503}
export CUDA_VISIBLE_DEVICES=0,1,2,3

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
export NCCL_IB_TIMEOUT=22
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_DIR}"

# -------- Resume: prefer latest checkpoint in OUTPUT_DIR --------
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [ -n "${LATEST_CKPT}" ] && [ -d "${LATEST_CKPT}" ]; then
    echo "RESUME: loading student from ${LATEST_CKPT}"
    MODEL_NAME_OR_PATH="${LATEST_CKPT}"
    for f in preprocessor_config.json video_preprocessor_config.json; do
        if [ ! -f "${LATEST_CKPT}/${f}" ] && [ -f "${SCRATCH_ROOT}/Qwen3-VL-2B-Instruct/${f}" ]; then
            cp -v "${SCRATCH_ROOT}/Qwen3-VL-2B-Instruct/${f}" "${LATEST_CKPT}/${f}"
        fi
    done
else
    echo "FRESH: starting GRACE from ${MODEL_NAME_OR_PATH}"
fi

for p in "${MODEL_NAME_OR_PATH}" "${TEACHER_MODEL_PATH}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: path does not exist: ${p}" >&2
        exit 1
    fi
done

echo "=============================="
echo "Qwen3-VL GRACE  (W${QAT_BITS}, group=${QAT_GROUP_SIZE})"
echo "=============================="
echo "STUDENT = ${MODEL_NAME_OR_PATH}"
echo "TEACHER = ${TEACHER_MODEL_PATH}"
echo "DATASETS= ${DATASETS}  (root=${SHAREGPT4V_ROOT})"
echo "OUTPUT  = ${OUTPUT_DIR}"
echo "NODES=${SLURM_NNODES}, Total GPUs=$((SLURM_NNODES * 4))"
echo "DKD: T=${DKD_TEMPERATURE} a=${DKD_ALPHA} b=${DKD_BETA} | RCKA: w=${RCKA_WEIGHT} L=${RCKA_LAYER}"
echo "IB : tau=${IB_TAU} eta=${IB_ETA} beta0=${IB_BETA_INIT} beta=[${IB_BETA_MIN},${IB_BETA_MAX}]"
echo "Warmup: dkd=${DKD_WARMUP_STEPS} rcka=${RCKA_WARMUP_STEPS}"
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
        -m qwenvl.train.train_qwen_grace \
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
        --per_device_eval_batch_size "${PER_DEVICE_BATCH}" \
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
        --report_to wandb \
        --qat_enable True \
        --qat_bits "${QAT_BITS}" \
        --qat_group_size "${QAT_GROUP_SIZE}" \
        --qat_scale_lr "${QAT_SCALE_LR}" \
        --qat_save_quantized True \
        --grace_enable True \
        --teacher_model_path "${TEACHER_MODEL_PATH}" \
        --dkd_temperature "${DKD_TEMPERATURE}" \
        --dkd_alpha "${DKD_ALPHA}" \
        --dkd_beta "${DKD_BETA}" \
        --rcka_weight "${RCKA_WEIGHT}" \
        --rcka_layer "${RCKA_LAYER}" \
        --ib_tau "${IB_TAU}" \
        --ib_eta "${IB_ETA}" \
        --ib_beta_init "${IB_BETA_INIT}" \
        --ib_beta_min "${IB_BETA_MIN}" \
        --ib_beta_max "${IB_BETA_MAX}" \
        --dkd_warmup_steps "${DKD_WARMUP_STEPS}" \
        --rcka_warmup_steps "${RCKA_WARMUP_STEPS}"

echo "Qwen3-VL GRACE training complete (output at ${OUTPUT_DIR})."
