#!/bin/bash

# MMMU Inference Script (Thinking Model)
# This script runs inference on the MMMU dataset using vLLM with thinking mode parameters

python run_mmmu.py infer \
    --model-path /path/to/Qwen3-VL-Thinking \
    --data-dir /path/to/mmmu_data \
    --dataset MMMU_DEV_VAL \
    --output-file results/mmmu_dev_val_predictions_thinking.jsonl \
    --max-new-tokens 40960 \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 0.0 \
    --tensor-parallel-size 4
