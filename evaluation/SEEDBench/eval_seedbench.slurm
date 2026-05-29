#!/bin/bash
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=qwen3vl_seed
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

cd "${HOME}"
module purge
module load profile/deeplrn
module load cuda/12.3
module load gcc/12.2.0
export PATH="${HOME}/python3.11/bin:${PATH}"
export PYTHONPATH=""
source "${HOME}/qwen3vl/bin/activate"

EVAL_DIR=${EVAL_DIR:-"${HOME}/Qwen2-VL/evaluation/SEEDBench"}
SCRATCH_ROOT=${SCRATCH_ROOT:-"/leonardo_scratch/large/userexternal/ychen003"}

MODEL_PATH=${MODEL_PATH:-"${SCRATCH_ROOT}/checkpoints/qwen3vl-2b-grace-w4g128"}
SEED_ROOT=${SEED_ROOT:-"${SCRATCH_ROOT}/ShareGPT4V/data/SEED-Bench"}
JSON_PATH=${JSON_PATH:-"${SEED_ROOT}/SEED-Bench.json"}
IMAGE_DIR=${IMAGE_DIR:-"${SEED_ROOT}/SEED-Bench-image"}

OUT_DIR=${OUT_DIR:-"${SCRATCH_ROOT}/eval_results/seedbench"}
TAG=$(basename "${MODEL_PATH}")
OUTPUT=${OUTPUT:-"${OUT_DIR}/${TAG}_img.json"}
mkdir -p "${OUT_DIR}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC=${NPROC:-4}

cd "${EVAL_DIR}"

echo "=============================="
echo "SEED-Bench-IMG eval"
echo "  EVAL_DIR  = ${EVAL_DIR}"
echo "  MODEL     = ${MODEL_PATH}"
echo "  JSON_PATH = ${JSON_PATH}"
echo "  IMAGE_DIR = ${IMAGE_DIR}"
echo "  OUTPUT    = ${OUTPUT}"
echo "  NPROC     = ${NPROC}"
echo "=============================="

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    eval_seedbench.py \
    --model_path "${MODEL_PATH}" \
    --json_path "${JSON_PATH}" \
    --image_dir "${IMAGE_DIR}" \
    --output "${OUTPUT}" \
    --attn flash_attention_2 \
    --max_new_tokens 8

echo "Done. Summary written to ${OUTPUT}"
