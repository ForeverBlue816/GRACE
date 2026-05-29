#!/bin/bash

# MathVision Evaluation Script (Thinking Model)
# This script evaluates the inference results from thinking model using GPT-4o

python run_mathv.py eval \
    --data-dir /path/to/mathvision_data \
    --input-file results/mathvision_predictions_thinking.jsonl \
    --output-file results/mathvision_eval_results_thinking.csv \
    --dataset MathVision \
    --eval-model gpt-4o-2024-05-13 \
    --api-type dash \
    --nproc 16
