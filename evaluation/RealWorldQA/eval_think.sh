#!/bin/bash

# RealWorldQA Evaluation Script (Thinking Model)
# This script evaluates the inference results from thinking model using rule-based and optionally model-based extraction

python run_realworldqa.py eval \
    --data-dir /path/to/data \
    --input-file results/RealWorldQA_results_thinking.jsonl \
    --output-file results/RealWorldQA_evaluation_thinking.csv \
    --dataset RealWorldQA \
    --eval-model qwen-plus \
    --api-type dash \
    --nproc 4

