# RealWorldQA Benchmark Evaluation

This directory contains the implementation for evaluating vision-language models on the RealWorldQA benchmark using vLLM for high-speed inference.

## Overview

RealWorldQA is a real-world visual question answering benchmark containing 700+ high-quality VQA samples covering various real-world scenarios. This implementation provides:

- **High-speed inference** using vLLM with automatic batch optimization
- **Two-stage evaluation** using rule-based extraction with optional LLM-based fallback
- **Automatic dataset download** from OpenCompass
- **Modular code structure** for easy maintenance and extension

## Project Structure

```
RealWorldQA/
├── run_realworldqa.py    # Main script for inference and evaluation
├── dataset_utils.py       # Dataset loading and preprocessing utilities
├── eval_utils.py          # Evaluation logic and answer extraction
├── common_utils.py        # Common utilities for image processing, file I/O
├── infer_instruct.sh      # Inference script for instruct models
├── infer_think.sh         # Inference script for thinking models
├── eval_instruct.sh       # Evaluation script for instruct model results
├── eval_think.sh          # Evaluation script for thinking model results
├── requirements.txt       # Python dependencies
└── README.md             # This file
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

For optional LLM-based evaluation, you need to set up API credentials:

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

### Data Preparation

The RealWorldQA dataset is stored in TSV format and will be **automatically downloaded** on first run from:
```
https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv
```

**Directory structure after download:**
```
${DATA_DIR}/
├── RealWorldQA.tsv          # Main data file (auto-downloaded)
└── images/
    └── RealWorldQA/         # Decoded image files
```

**Setting data path:**
- Option 1: Environment variable `export LMUData="/path/to/data"`
- Option 2: Use `--data-dir` argument in commands

## Quick Start

### 1. Inference

Run inference on the RealWorldQA dataset using an instruct model:

```bash
bash infer_instruct.sh
```

Or customize the inference:

```bash
python run_realworldqa.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/data \
    --dataset RealWorldQA \
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

Evaluate the inference results:

```bash
bash eval_instruct.sh
```

Or customize the evaluation:

```bash
python run_realworldqa.py eval \
    --data-dir /path/to/data \
    --input-file results/predictions.jsonl \
    --output-file results/evaluation.csv \
    --dataset RealWorldQA \
    --eval-model gpt-3.5-turbo-0125 \
    --api-type dash \
    --nproc 4
```

## Detailed Usage

### Inference Mode

**Basic Arguments:**
- `--model-path`: Path to the Qwen3-VL model directory (required)
- `--data-dir`: Directory to store/load RealWorldQA dataset (required)
- `--dataset`: Dataset name (default: `RealWorldQA`)
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
- `--min-pixels`: Minimum pixels for image (default: 768×28×28 ≈ 600K pixels)
- `--max-pixels`: Maximum pixels for image (default: 5120×28×28 ≈ 4M pixels)

### Evaluation Mode

**Basic Arguments:**
- `--data-dir`: Directory containing RealWorldQA dataset (required)
- `--input-file`: Inference results file in JSONL format (required)
- `--output-file`: Path to save evaluation results in CSV format (required)
- `--dataset`: Dataset name, must match inference (default: `RealWorldQA`)

**Judge Model Arguments:**
- `--eval-model`: Judge model name (default: None, uses rule-based only)
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
  "question_id": 0,
  "annotation": {
    "index": "0",
    "question": "What is shown in the image?",
    "A": "Cat",
    "B": "Dog",
    "C": "Bird",
    "D": "Fish",
    "answer": "A"
  },
  "task": "RealWorldQA",
  "result": {
    "gen": "The correct answer is A",
    "gen_raw": "Raw model output including thinking process"
  },
  "messages": [...]
}
```

### Evaluation Output

The evaluation script generates two files:

1. **CSV file** (`*_evaluation.csv`): Detailed evaluation results
   - Columns: `index`, `question`, `prediction`, `extracted_answer`, `extraction_method`, `extraction_success`, `gt`, `hit`

2. **JSON file** (`*_evaluation_acc.json`): Accuracy statistics
   ```json
   {
     "overall_accuracy": 0.7234,
     "task_samples": 765,
     "correct": 553,
     "total": 765
   }
   ```

## Model-Specific Configurations

### Instruct Models (e.g., Qwen3-VL-2B-Instruct, Qwen3-VL-7B-Instruct)

Use standard parameters for balanced performance:

```bash
--max-new-tokens 32768
--temperature 0.7
--top-p 0.8
--top-k 20
--repetition-penalty 1.0
--presence-penalty 1.5
```

### Thinking Models (e.g., Qwen3-VL-2B-Thinking)

Use adjusted parameters for deeper reasoning:

```bash
--max-new-tokens 32768
--temperature 0.6
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
   - 2B model: 1 GPU recommended
   - 7B model: 1-2 GPUs
   - 14B+ model: 2-4 GPUs

4. **Context Length**: Reduce `--max-model-len` if memory is limited:
   - 128000: Default, works well for most cases
   - 64000: Reduces memory usage by ~40%

5. **Evaluation Speed**: Omit `--eval-model` to use rule-based extraction only (faster, ~70-80% success rate)

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.7

# Or reduce context length
--max-model-len 64000

# Or reduce image resolution
--max-pixels 1003520  # 1280×28×28
```

**2. vLLM Multiprocessing Issues**
The code automatically sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`. If you still encounter issues:
```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

**3. Evaluation API Errors**
- If you don't need LLM-based extraction, omit `--eval-model` (rule-based only)
- If using LLM extraction, verify API credentials are set correctly
- Check API endpoint connectivity
- Increase `--nproc` value if rate-limited (up to 32)

**4. Dataset Download Issues**
The dataset is automatically downloaded from:
```
https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv
```
If download fails, manually download and place in `${DATA_DIR}/RealWorldQA.tsv`

**5. Import Errors**
Ensure all required files exist in the RealWorldQA directory:
```bash
ls common_utils.py dataset_utils.py eval_utils.py run_realworldqa.py
```

## Advanced Usage

### Custom Image Resolution

Modify resolution parameters in the inference command:

```bash
python run_realworldqa.py infer \
    --min-pixels 393216      # 512×28×28
    --max-pixels 1003520     # 1280×28×28
    ...
```

### Evaluation Without LLM

Use rule-based extraction only (faster, no API calls):

```bash
python run_realworldqa.py eval \
    --input-file results/predictions.jsonl \
    --output-file results/evaluation.csv
    # No --eval-model specified
```

### Debug Mode

Process only first N samples for testing:

```bash
DEBUG_SAMPLE_SIZE=10 python run_realworldqa.py infer ...
```

## Citation

If you use this code or the RealWorldQA benchmark, please cite:

```bibtex
@misc{realworldqa2024,
  title        = {RealWorldQA: A Benchmark for Real-World Spatial Understanding},
  author       = {{xAI}},
  year         = {2024},
  howpublished = {\url{https://huggingface.co/datasets/xai-org/RealworldQA}},
  note         = {Accessed: 2025-04-26}
}
```

## License

This code is released under the same license as the Qwen3-VL model.

## Support

For issues and questions:
- GitHub Issues: [Qwen3-VL Repository](https://github.com/QwenLM/Qwen3-VL)
- Documentation: See inline code comments and docstrings
