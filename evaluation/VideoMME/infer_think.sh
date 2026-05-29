#!/bin/bash

# VideoMME Inference Script (Thinking Model)
# This script runs inference on the VideoMME dataset using vLLM with thinking mode parameters

# For testing with first 10 samples, add: --max-samples 10
# For full evaluation, remove --max-samples parameter

python run_videomme.py infer \
    --model-path /path/to/Qwen3-VL-Thinking \
    --data-dir /path/to/VideoMME \
    --duration short \
    --output-file results/videomme_short_wo_subtitle_predictions_thinking.jsonl \
    --max-new-tokens 32768 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 0.0 \
    --fps 2 \
    --min-pixels 3584 \
    --max-pixels 401408 \
    --min-frames 4 \
    --max-frames 512 \
    --total-pixels 19267584 \
    --tensor-parallel-size 4

