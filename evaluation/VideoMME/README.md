# VideoMME Benchmark Evaluation

This directory contains the implementation for evaluating vision-language models on the VideoMME (Video Multi-Modal Evaluation) benchmark using vLLM for high-speed inference.

## Overview

The VideoMME benchmark evaluates models on video understanding tasks with multiple-choice questions across various domains. This implementation provides:

- **High-speed inference** using vLLM with automatic batch optimization
- **Flexible evaluation** using GPT-based judge models
- **Support for thinking models** with extended reasoning
- **Modular code structure** for easy maintenance and extension
- **Video processing** with flexible frame sampling and subtitle support

## Project Structure

```
VideoMME/
├── run_videomme.py       # Main script for inference and evaluation
├── dataset_utils.py      # Dataset loading and video processing utilities
├── eval_utils.py         # Evaluation logic and judge model wrappers
├── infer_instruct.sh     # Inference script for instruct models
├── infer_think.sh        # Inference script for thinking models
├── eval_instruct.sh      # Evaluation script for instruct model results
├── eval_think.sh         # Evaluation script for thinking model results
├── requirements.txt      # Python dependencies
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
- `decord` - Video reading and decoding library
- `pysubs2` - Subtitle parsing library
- `datasets` - HuggingFace datasets for data loading
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

### Data Preparation

1. Download the VideoMME dataset from the official source
2. Organize the data directory structure:

```
VideoMME/
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── subtitle/
│   ├── video1.srt
│   ├── video2.srt
│   └── ...
└── dataset files (automatically loaded via HuggingFace datasets)
```

## Quick Start

### 1. Inference

Run inference on VideoMME dataset using an instruct model:

```bash
bash infer_instruct.sh
```

Or customize the inference:

```bash
python run_videomme.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/VideoMME \
    --duration short \
    --output-file results/videomme_short_wo_subtitle_predictions.jsonl \
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
python run_videomme.py eval \
    --data-dir /path/to/VideoMME \
    --input-file results/videomme_short_wo_subtitle_predictions.jsonl \
    --output-file results/videomme_short_wo_subtitle_eval_results.csv \
    --eval-model gpt-3.5-turbo-0125 \
    --api-type dash \
    --nproc 4
```

## Detailed Usage

### Inference Mode

**Basic Arguments:**
- `--model-path`: Path to the Qwen3-VL model directory (required)
- `--data-dir`: Directory containing VideoMME dataset (required)
- `--duration`: Video duration type: `short`, `medium`, or `long` (default: `short`)
- `--output-file`: Path to save inference results in JSONL format (required)

**Video Processing Arguments:**
- `--fps`: Frames per second to extract (default: 2)
- `--min-pixels`: Minimum pixels per frame (default: 3584)
- `--max-pixels`: Maximum pixels per frame (default: 401408)
- `--min-frames`: Minimum number of frames to extract (default: 4)
- `--max-frames`: Maximum number of frames to extract (default: 512)
- `--total-pixels`: Total pixels across all frames (default: 19267584)

**Subtitle Arguments:**
- `--use-subtitle`: Enable subtitle integration (optional flag)
- `--sys-prompt`: Path to custom system prompt file (optional)

**vLLM Arguments:**
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (default: auto-detect)
- `--gpu-memory-utilization`: GPU memory utilization ratio, 0.0-1.0 (default: 0.9)
- `--max-model-len`: Maximum model context length (default: 128000)
- `--max-videos-per-prompt`: Maximum videos per prompt (default: 1)

**Generation Parameters:**
- `--max-new-tokens`: Maximum tokens to generate (default: 32768)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling (default: 0.8)
- `--top-k`: Top-k sampling (default: 20)
- `--repetition-penalty`: Repetition penalty (default: 1.0)
- `--presence-penalty`: Presence penalty to reduce repetition (default: 1.5)

**Advanced Options:**
- `--max-samples`: Process only first N samples for testing (optional)

### Evaluation Mode

**Basic Arguments:**
- `--data-dir`: Directory containing VideoMME dataset (required)
- `--input-file`: Inference results file in JSONL format (required)
- `--output-file`: Path to save evaluation results in CSV format (required)

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
  "question_id": "q001",
  "annotation": {
    "videoID": "video_001",
    "question": "What is happening in the video?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "A",
    "domain": "perception",
    "sub_category": "object_recognition"
  },
  "task": "VideoMME_short_wo_subtitle",
  "result": {
    "gen": "A",
    "gen_raw": "The answer is A because..."
  },
  "messages": [...]
}
```

### Evaluation Output

The evaluation script generates multiple files:

1. **CSV file** (`*_eval_results.csv`): Detailed results for each sample
   - Columns: `index`, `question`, `prediction`, `extracted_answer`, `extraction_method`, `gt`, `hit`, `domain`, `sub_category`

2. **TSV file** (`*_eval_results.tsv`): Same data in TSV format for compatibility

3. **JSON file** (`*_eval_results_acc.json`): Accuracy summary
   ```json
   {
     "overall_accuracy": 0.7234,
     "accuracy_by_domain": {
       "perception": 0.7543,
       "reasoning": 0.6891,
       "knowledge": 0.7321
     },
     "accuracy_by_subcategory": {
       "object_recognition": 0.8012,
       "spatial_reasoning": 0.6534,
       ...
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
--max-new-tokens 32768
--temperature 0.6
--top-p 0.95
--top-k 20
--repetition-penalty 1.0
--presence-penalty 0.0
```

**Note:** Thinking models output reasoning steps wrapped in `<think>...</think>` tags. The evaluation automatically extracts the final answer after `</think>`.

## Video Duration Guidelines

### Short Videos
- **Characteristics**: Typically < 2 minutes
- **Recommended frames**: 4-64 frames
- **FPS setting**: 2 fps
- **Example**: `--duration short --fps 2 --max-frames 512`

### Medium Videos
- **Characteristics**: Typically 2-15 minutes
- **Recommended frames**: 64-256 frames
- **FPS setting**: 1-2 fps
- **Example**: `--duration medium --fps 1 --max-frames 512`

### Long Videos
- **Characteristics**: Typically > 15 minutes
- **Recommended frames**: 256-512 frames
- **FPS setting**: 0.5-1 fps
- **Example**: `--duration long --fps 1 --max-frames 512`

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

5. **Frame Sampling**: Adjust frame parameters based on video length:
   - Short videos: Higher FPS (2-4), more frames
   - Long videos: Lower FPS (0.5-1), fewer frames per second
   - Use `--total-pixels` to control overall memory usage

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.7

# Or reduce context length
--max-model-len 64000

# Or reduce max frames
--max-frames 256
```

**2. Video Loading Errors**
- Check video file integrity and codec compatibility
- Install ffmpeg: `apt-get install ffmpeg` or `brew install ffmpeg`
- Verify video paths in dataset

**3. vLLM Multiprocessing Issues**
The code automatically sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`. If you still encounter issues:
```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

**4. Evaluation API Errors**
- Verify API credentials are set correctly
- Check API endpoint connectivity
- Reduce `--nproc` value if rate-limited
- Increase `--nproc` for faster evaluation (up to 32)

**5. Subtitle Loading Issues**
- Ensure subtitle files match video IDs
- Check subtitle directory structure (`subtitle/*.srt`)
- Use `--use-subtitle` only when subtitles are available

**6. Dataset Download Issues**
The dataset is automatically loaded via HuggingFace datasets. If loading fails:
- Check internet connectivity
- Verify HuggingFace datasets installation
- Manually download and place dataset files in `--data-dir`

## Advanced Usage

### Using Subtitles

Enable subtitle integration for videos with available subtitles:

```bash
python run_videomme.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/VideoMME \
    --duration short \
    --use-subtitle \
    --output-file results/videomme_short_w_subtitle_predictions.jsonl
```

### Custom System Prompt

Provide a custom system prompt for specialized tasks:

```bash
echo "You are a video understanding expert. Analyze the video carefully and select the best answer." > sys_prompt.txt

python run_videomme.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/VideoMME \
    --duration medium \
    --sys-prompt sys_prompt.txt \
    --output-file results/videomme_medium_predictions.jsonl
```

### Processing Different Video Durations

**Short videos:**
```bash
python run_videomme.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/VideoMME \
    --duration short \
    --fps 2 \
    --max-frames 512 \
    --output-file results/videomme_short_predictions.jsonl
```

**Medium videos:**
```bash
python run_videomme.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/VideoMME \
    --duration medium \
    --fps 1 \
    --max-frames 512 \
    --output-file results/videomme_medium_predictions.jsonl
```

**Long videos:**
```bash
python run_videomme.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/VideoMME \
    --duration long \
    --fps 1 \
    --max-frames 512 \
    --max-model-len 64000 \
    --output-file results/videomme_long_predictions.jsonl
```

### Custom Video Resolution

Edit frame resolution parameters for different quality/memory trade-offs:

```python
# In run_videomme.py or via command line:
--min-pixels 3584        # Minimum pixels per frame (~224x224)
--max-pixels 401408      # Maximum pixels per frame (~512x512)
--total-pixels 19267584  # Total pixels across all frames
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

Enable debug mode to process only first 5-10 samples:

```bash
python run_videomme.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/VideoMME \
    --duration short \
    --max-samples 10 \
    --output-file results/debug_predictions.jsonl
```

## Evaluation Methodology

### Two-Stage Answer Extraction

1. **Rule-Based Extraction** (Fast, no API needed)
   - Direct option extraction (A/B/C/D) from model output
   - Pattern matching for clear answer indicators
   - Success rate: ~70-90% depending on model output quality

2. **LLM-Based Extraction** (Fallback, requires API)
   - Use judge model (GPT-3.5/GPT-4) to match answer to options
   - Only triggered when rule-based extraction fails
   - Provides robust extraction for complex or ambiguous responses

### Scoring

- Binary accuracy: 1 if extracted answer matches ground truth, 0 otherwise
- Aggregated by overall, domain, and sub-category
- Detailed per-sample results saved in CSV/TSV format

## Citation

If you use this code or the VideoMME benchmark, please cite:

```bibtex
@article{videomme2024,
  title={VideoMME: A Large-Scale Video Understanding Benchmark},
  author={},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This code is released under the same license as the Qwen3-VL model.

## Support

For issues and questions:
- GitHub Issues: [Qwen3-VL Repository](https://github.com/QwenLM/Qwen3-VL)
- Documentation: See inline code comments and docstrings
