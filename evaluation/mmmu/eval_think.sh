#!/bin/bash

# MMMU Evaluation Script (Thinking Model)
# This script evaluates the inference results from thinking model using a judge model

python run_mmmu.py eval \
    --data-dir /path/to/mmmu_data \
    --input-file results/mmmu_dev_val_predictions_thinking.jsonl \
    --output-file results/mmmu_dev_val_eval_results_thinking.csv \
    --dataset MMMU_DEV_VAL \
    --eval-model gpt-3.5-turbo-0125 \
    --api-type dash \
    --nproc 16