#!/bin/bash
#SBATCH -A <your_slurm_account>
#SBATCH -p <your_partition>
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=qwen3vl_mmmu
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

EVAL_DIR=${EVAL_DIR:-"${HOME}/Qwen2-VL/evaluation/MMMU"}
SCRATCH_ROOT=${SCRATCH_ROOT:-"/path/to/scratch"}

MODEL_PATH=${MODEL_PATH:-"${SCRATCH_ROOT}/checkpoints/qwen3vl-2b-grace-w4g128"}
DATA_DIR=${DATA_DIR:-"${SCRATCH_ROOT}/ShareGPT4V/data/MMMU"}
SPLIT=${SPLIT:-"validation"}

OUT_DIR=${OUT_DIR:-"${SCRATCH_ROOT}/eval_results/mmmu"}
TAG=$(basename "${MODEL_PATH}")
OUTPUT=${OUTPUT:-"${OUT_DIR}/${TAG}_${SPLIT}.json"}
mkdir -p "${OUT_DIR}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC=${NPROC:-4}

cd "${EVAL_DIR}"

echo "=============================="
echo "MMMU eval"
echo "  EVAL_DIR = ${EVAL_DIR}"
echo "  MODEL    = ${MODEL_PATH}"
echo "  DATA_DIR = ${DATA_DIR}"
echo "  SPLIT    = ${SPLIT}"
echo "  OUTPUT   = ${OUTPUT}"
echo "  NPROC    = ${NPROC}"
echo "=============================="

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    eval_mmmu.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --split "${SPLIT}" \
    --output "${OUTPUT}" \
    --attn flash_attention_2 \
    --max_new_tokens 32

echo "Done. Summary written to ${OUTPUT}"
