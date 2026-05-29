#!/bin/bash

# RealWorldQA Evaluation Script (Instruct Model)
# This script evaluates the inference results using rule-based and optionally model-based extraction

python run_realworldqa.py eval \
    --data-dir /path/to/data \
    --input-file results/RealWorldQA_results.jsonl \
    --output-file results/RealWorldQA_evaluation.csv \
    --dataset RealWorldQA \
    --eval-model gpt-3.5-turbo-0125 \
    --api-type dash \
    --nproc 4

