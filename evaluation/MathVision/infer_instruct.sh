#!/bin/bash

# MathVision Inference Script (Instruct Model)
# This script runs inference on the MathVision dataset using vLLM

python run_mathv.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/mathvision_data \
    --dataset MathVision \
    --output-file results/mathvision_predictions.jsonl \
    --max-new-tokens 32768 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5
    # --num-samples 100