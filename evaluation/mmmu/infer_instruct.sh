#!/bin/bash

# MMMU Inference Script (Instruct Model)
# This script runs inference on the MMMU dataset using vLLM

python run_mmmu.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/mmmu_data \
    --dataset MMMU_DEV_VAL \
    --output-file results/mmmu_dev_val_predictions.jsonl \
    --max-new-tokens 32768 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5
