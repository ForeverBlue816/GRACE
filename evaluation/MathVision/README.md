# MathVision Benchmark Evaluation

This directory contains the implementation for evaluating vision-language models on the MathVision benchmark using vLLM for high-speed inference.

## Overview

MathVision is a mathematical visual reasoning benchmark that evaluates models' ability to solve mathematical problems based on visual information. This implementation provides:

- **High-speed inference** using vLLM with automatic batch optimization
- **Two-stage evaluation** using rule-based and GPT-4o-based answer extraction
- **Support for thinking models** with extended reasoning capabilities
- **Modular code structure** for easy maintenance and extension

## Project Structure

```
MathVision/
├── run_mathv.py          # Main script for inference and evaluation
├── dataset_utils.py      # Dataset loading and preprocessing utilities
├── eval_utils.py         # Evaluation logic and answer extraction
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
- `latex2sympy2` - LaTeX to symbolic math conversion (optional)
- `openpyxl` - Excel file handling
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

Run inference on MathVision dataset using an instruct model:

```bash
bash infer_instruct.sh
```

Or customize the inference:

```bash
python run_mathv.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/data \
    --dataset MathVision \
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

Evaluate the inference results using GPT-4o as a judge:

```bash
bash eval_instruct.sh
```

Or customize the evaluation:

```bash
python run_mathv.py eval \
    --data-dir /path/to/data \
    --input-file results/predictions.jsonl \
    --output-file results/evaluation.csv \
    --dataset MathVision \
    --eval-model gpt-4o-2024-05-13 \
    --api-type dash \
    --nproc 16
```

## Detailed Usage

### Inference Mode

**Basic Arguments:**
- `--model-path`: Path to the Qwen3-VL model directory (required)
- `--data-dir`: Directory to store/load MathVision dataset (required)
- `--dataset`: Dataset name (default: `MathVision`)
  - `MathVision`: Full dataset with ~3,000 samples
  - `MathVision_MINI`: Mini version for quick testing
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
- `--cot-prompt`: Custom CoT prompt (default: " Let's think step by step.")
- `--num-samples`: Number of samples to process (optional, for testing)

### Evaluation Mode

**Basic Arguments:**
- `--data-dir`: Directory containing MathVision dataset (required)
- `--input-file`: Inference results file in JSONL format (required)
- `--output-file`: Path to save evaluation results in CSV format (required)
- `--dataset`: Dataset name, must match inference (default: `MathVision`)

**Judge Model Arguments:**
- `--eval-model`: Judge model name (default: `gpt-4o`)
  - Options: `gpt-4o`, `gpt-4o-2024-05-13`, `gpt-3.5-turbo-0125`, etc.
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
    "question": "What is the area of the triangle?",
    "answer": "12",
    "category": "Geometry",
    "choices": "[]",
    ...
  },
  "task": "MathVision",
  "result": {
    "gen": "The final answer",
    "gen_raw": "Raw model output including thinking process"
  },
  "messages": [...]
}
```

### Evaluation Output

The evaluation script generates multiple files:

1. **Intermediate results** (`*_eval_results.xlsx`): Raw predictions with metadata
2. **Detailed evaluation** (`*_eval_results_eval.xlsx`): Results with extracted answers
   - Columns: `index`, `question`, `prediction`, `answer`, `res` (extracted), `log`, `extract_model`, `extract_flag`, `category`
3. **Score summary** (`*_eval_results_eval_score.csv`): Accuracy by category

Example score summary:
```
Subject         | tot | prefetch | hit | prefetch_rate | acc
----------------|-----|----------|-----|---------------|------
Overall         | 3000| 2400     | 2100| 80.0          | 70.0
Algebra         | 800 | 640      | 560 | 80.0          | 70.0
Geometry        | 750 | 600      | 525 | 80.0          | 70.0
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

### Thinking Models (e.g., Qwen3-VL-4B-Thinking, Qwen3-VL-30B-Thinking)

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

5. **Image Resolution**: MathVision uses optimized resolution (768×28×28 to 5120×28×28)
   - Lower min_pixels for faster processing
   - Higher max_pixels for better accuracy on complex diagrams

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
- Monitor rate limits
- Increase `--nproc` value if rate-limited (up to 32)

**4. Dataset Download Issues**
The dataset is automatically downloaded from:
```
https://opencompass.openxlab.space/utils/VLMEval/MathVision.tsv
```
If download fails, manually download and place in `--data-dir`.

**5. Excel Export Errors**
The code automatically removes illegal Excel characters. If you still encounter issues:
- Check `clean_for_excel()` function in `run_mathv.py`
- Ensure `openpyxl` is installed

**6. LaTeX Conversion Errors**
Install `latex2sympy2` for better LaTeX support:
```bash
pip install latex2sympy2
```

## Advanced Usage

### Custom Image Resolution

Edit `run_mathv.py` to modify image resolution:

```python
MIN_PIXELS = 768*28*28   # ~0.6M pixels
MAX_PIXELS = 5120*28*28  # ~4M pixels
```

### Custom Evaluation Prompts

The evaluation uses in-context examples defined in `eval_utils.py`:
- Edit `get_gpt4_ICE()` to customize examples
- Edit `build_mathv_gpt4_prompt()` to modify prompt structure

### Testing with Limited Samples

Use `--num-samples` for quick testing:

```bash
python run_mathv.py infer \
    --model-path /path/to/model \
    --data-dir /path/to/data \
    --dataset MathVision \
    --output-file results/test.jsonl \
    --num-samples 100
```

### Debugging

Enable debug mode for detailed logs:

```bash
DEBUG=true python run_mathv.py eval ...
```

This processes only the first 5 samples in single-threaded mode.

## Citation

If you use this code or the MathVision benchmark, please cite:

```bibtex
@article{mathvision,
  title={Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset}, 
  author={Ke Wang and Junting Pan and Weikang Shi and Zimu Lu and Mingjie Zhan and Hongsheng Li},
  journal={arXiv:2402.14804},
  year={2024}
}
```

## License

This code is released under the same license as the Qwen3-VL model.

## Support

For issues and questions:
- GitHub Issues: [Qwen3-VL Repository](https://github.com/QwenLM/Qwen3-VL)
- Documentation: See inline code comments and docstrings
