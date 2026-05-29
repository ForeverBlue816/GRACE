# MMMU Benchmark Evaluation

This directory contains the implementation for evaluating vision-language models on the MMMU (Massive Multi-discipline Multimodal Understanding) benchmark using vLLM for high-speed inference.

## Overview

The MMMU benchmark evaluates models across diverse disciplines with multi-modal questions. This implementation provides:

- **High-speed inference** using vLLM with automatic batch optimization
- **Flexible evaluation** using GPT-based judge models
- **Support for thinking models** with extended reasoning
- **Modular code structure** for easy maintenance and extension

## Project Structure

```
mmmu/
├── run_mmmu.py           # Main script for inference and evaluation
├── dataset_utils.py      # Dataset loading and preprocessing utilities
├── eval_utils.py         # Evaluation logic and judge model wrappers
├── common_utils.py       # Common utilities for image processing, file I/O
├── infer_instruct.sh     # Inference script for instruct models
├── infer_think.sh        # Inference script for thinking models
├── eval_instruct.sh      # Evaluation script for instruct model results
├── eval_think.sh         # Evaluation script for thinking model results
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `vllm` - High-speed LLM inference engine
- `transformers` - HuggingFace transformers
- `qwen_vl_utils` - Qwen VL utilities for vision processing
- `pandas`, `numpy` - Data processing
- `Pillow` - Image processing
- `requests` - API calls for evaluation

### Environment Variables

For evaluation, you need to set up API credentials for the judge model:

**Option 1: DashScope API (Recommended)**
```bash
export CHATGPT_DASHSCOPE_API_KEY="your-api-key"
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
```

**Option 2: Custom OpenAI-compatible API**
```bash
export MIT_SPIDER_TOKEN="your-api-key"
export MIT_SPIDER_URL="your-api-endpoint"
```

## Quick Start

### 1. Inference

Run inference on MMMU dataset using an instruct model:

```bash
bash infer_instruct.sh
```

Or customize the inference:

```bash
python run_mmmu.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/data \
    --dataset MMMU_DEV_VAL \
    --output-file results/predictions.jsonl \
    --max-new-tokens 32768 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5
```

For thinking models with extended reasoning:

```bash
bash infer_think.sh
```

### 2. Evaluation

Evaluate the inference results using a judge model:

```bash
bash eval_instruct.sh
```

Or customize the evaluation:

```bash
python run_mmmu.py eval \
    --data-dir /path/to/data \
    --input-file results/predictions.jsonl \
    --output-file results/evaluation.csv \
    --dataset MMMU_DEV_VAL \
    --eval-model gpt-3.5-turbo-0125 \
    --api-type dash \
    --nproc 16
```

## Detailed Usage

### Inference Mode

**Basic Arguments:**
- `--model-path`: Path to the Qwen3-VL model directory (required)
- `--data-dir`: Directory to store/load MMMU dataset (required)
- `--dataset`: Dataset name, default: `MMMU_DEV_VAL`
- `--output-file`: Path to save inference results in JSONL format (required)

**vLLM Arguments:**
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (default: auto-detect)
- `--gpu-memory-utilization`: GPU memory utilization ratio, 0.0-1.0 (default: 0.9)
- `--max-model-len`: Maximum model context length (default: 128000)
- `--max-images-per-prompt`: Maximum images per prompt (default: 10)

**Generation Parameters:**
- `--max-new-tokens`: Maximum tokens to generate (default: 32768)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling (default: 0.8)
- `--top-k`: Top-k sampling (default: 20)
- `--repetition-penalty`: Repetition penalty (default: 1.0)
- `--presence-penalty`: Presence penalty to reduce repetition (default: 1.5)

**Advanced Options:**
- `--use-cot`: Enable Chain-of-Thought prompting for better reasoning
- `--cot-prompt`: Custom CoT prompt (optional)

### Evaluation Mode

**Basic Arguments:**
- `--data-dir`: Directory containing MMMU dataset (required)
- `--input-file`: Inference results file in JSONL format (required)
- `--output-file`: Path to save evaluation results in CSV format (required)
- `--dataset`: Dataset name, must match inference (default: `MMMU_DEV_VAL`)

**Judge Model Arguments:**
- `--eval-model`: Judge model name (default: `gpt-3.5-turbo-0125`)
  - Options: `gpt-3.5-turbo-0125`, `gpt-4-0125-preview`, `gpt-4o`, etc.
- `--api-type`: API service type (default: `dash`)
  - `dash`: DashScope API (Alibaba Cloud)
  - `mit`: Custom OpenAI-compatible API
- `--nproc`: Number of parallel workers for evaluation (default: 4)

## Output Files

### Inference Output

The inference script generates a JSONL file where each line contains:

```json
{
  "question_id": 123,
  "annotation": {
    "index": 123,
    "question": "What is shown in the image?",
    "A": "Option A",
    "B": "Option B",
    "answer": "A",
    ...
  },
  "task": "MMMU_DEV_VAL",
  "result": {
    "gen": "The final answer",
    "gen_raw": "Raw model output including thinking process"
  },
  "messages": [...]
}
```

### Evaluation Output

The evaluation script generates two files:

1. **CSV file** (`*_eval_results.csv`): Detailed results for each sample
   - Columns: `index`, `question`, `prediction`, `extracted_answer`, `extraction_method`, `gt`, `hit`, `split`

2. **JSON file** (`*_eval_results_acc.json`): Accuracy summary
   ```json
   {
     "overall_accuracy": 0.7234,
     "accuracy_by_split": {
       "validation": 0.7234
     }
   }
   ```

## Model-Specific Configurations

### Instruct Models (e.g., Qwen3-VL-2B-Instruct, Qwen3-VL-30B-Instruct)

Use standard parameters for balanced performance:

```bash
--max-new-tokens 32768
--temperature 0.7
--top-p 0.8
--top-k 20
--repetition-penalty 1.0
--presence-penalty 1.5
```

### Thinking Models (e.g., Qwen3-VL-4B-Thinking)

Use extended parameters for deeper reasoning:

```bash
--max-new-tokens 40960
--temperature 1.0
--top-p 0.95
--top-k 20
--repetition-penalty 1.0
--presence-penalty 0.0
```

**Note:** Thinking models output reasoning steps wrapped in `<think>...</think>` tags. The evaluation automatically extracts the final answer after `</think>`.

## Performance Tips

1. **GPU Memory**: Adjust `--gpu-memory-utilization` based on your GPU:
   - 0.9: Recommended for most cases
   - 0.95: For maximum throughput (may cause OOM)
   - 0.7-0.8: If experiencing OOM errors

2. **Batch Size**: vLLM automatically optimizes batch size based on available memory

3. **Tensor Parallelism**: Use `--tensor-parallel-size` for large models:
   - 2B/4B models: 1-2 GPUs
   - 7B/14B models: 2-4 GPUs
   - 30B+ models: 4-8 GPUs

4. **Context Length**: Reduce `--max-model-len` if memory is limited:
   - 128000: Default, works well for most cases
   - 64000: Reduces memory usage by ~40%

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.7

# Or reduce context length
--max-model-len 64000
```

**2. vLLM Multiprocessing Issues**
The code automatically sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`. If you still encounter issues:
```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

**3. Evaluation API Errors**
- Verify API credentials are set correctly
- Check API endpoint connectivity
- Increase `--nproc` value if rate-limited (up to 32)

**4. Dataset Download Issues**
The dataset is automatically downloaded from:
```
https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv
```
If download fails, manually download and place in `--data-dir`.

## Advanced Usage

### Custom Image Resolution

Edit `run_mmmu.py` to modify image resolution:

```python
MIN_PIXELS = 1280*28*28  # ~1M pixels
MAX_PIXELS = 5120*28*28  # ~4M pixels
```

### Custom Evaluation Logic

The evaluation uses a two-stage approach:
1. **Rule-based extraction**: Fast pattern matching for clear answers
2. **Model-based extraction**: GPT judge for ambiguous answers

To customize, edit `eval_utils.py`:
- `can_infer_option()`: Modify option extraction rules
- `can_infer_text()`: Modify text matching logic
- `build_prompt()`: Customize judge prompt

### Debugging

Enable debug mode to process only 5 samples:

```bash
DEBUG=true python run_mmmu.py eval ...
```

## Citation

If you use this code or the MMMU benchmark, please cite:

```bibtex
@article{yue2023mmmu,
  title={Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi},
  author={Yue, Xiang and Ni, Yuansheng and Zhang, Kai and Zheng, Tianyu and Liu, Ruoqi and Zhang, Ge and Stevens, Samuel and Jiang, Dongfu and Ren, Weiming and Sun, Yuxuan and others},
  journal={arXiv:2311.16502},
  year={2023}
}
```

## License

This code is released under the same license as the Qwen3-VL model.

## Support

For issues and questions:
- GitHub Issues: [Qwen3-VL Repository](https://github.com/QwenLM/Qwen3-VL)
- Documentation: See inline code comments and docstrings
