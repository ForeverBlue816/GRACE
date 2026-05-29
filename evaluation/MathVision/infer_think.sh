#!/bin/bash

# MathVision Inference Script (Thinking Model)
# This script runs inference on the MathVision dataset using vLLM with thinking mode parameters

python run_mathv.py infer \
    --model-path /path/to/Qwen3-VL-Thinking \
    --data-dir /path/to/mathvision_data \
    --dataset MathVision \
    --output-file results/mathvision_predictions_thinking.jsonl \
    --max-new-tokens 40960 \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 0.0
    # --num-samples 100