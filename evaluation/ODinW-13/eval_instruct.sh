#!/bin/bash

# ODinW Evaluation Script (Instruct Model)
# This script evaluates the inference results using COCO metrics

python run_odinw.py eval \
    --data-dir /path/to/odinw_data \
    --input-file results/odinw_predictions.jsonl \
    --output-file results/odinw_eval_results.json

