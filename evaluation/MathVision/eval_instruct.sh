#!/bin/bash

# MathVision Evaluation Script (Instruct Model)
# This script evaluates the inference results using GPT-4o

python run_mathv.py eval \
    --data-dir /path/to/mathvision_data \
    --input-file results/mathvision_predictions.jsonl \
    --output-file results/mathvision_eval_results.csv \
    --dataset MathVision \
    --eval-model gpt-4o-2024-05-13 \
    --api-type dash \
    --nproc 16
