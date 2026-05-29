#!/bin/bash

# VideoMME Inference Script (Instruct Model)
# This script runs inference on the VideoMME dataset using vLLM

# For testing with first 10 samples, add: --max-samples 10
# For full evaluation, remove --max-samples parameter

python run_videomme.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/VideoMME \
    --duration short \
    --output-file results/videomme_short_wo_subtitle_predictions.jsonl \
    --max-new-tokens 32768 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5 \
    --fps 2 \
    --min-pixels 3584 \
    --max-pixels 401408 \
    --min-frames 4 \
    --max-frames 512 \
    --total-pixels 19267584

