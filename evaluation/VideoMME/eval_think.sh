#!/bin/bash

# VideoMME Evaluation Script (Thinking Model)
# This script evaluates the inference results from thinking model using a judge model

python run_videomme.py eval \
    --data-dir /path/to/VideoMME \
    --input-file results/videomme_short_wo_subtitle_predictions_thinking.jsonl \
    --output-file results/videomme_short_wo_subtitle_eval_results_thinking.csv \
    --eval-model gpt-3.5-turbo-0125 \
    --api-type dash \
    --nproc 4

