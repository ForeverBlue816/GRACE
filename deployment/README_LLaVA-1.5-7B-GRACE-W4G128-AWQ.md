---
license: apache-2.0
language:
- en

library_name: transformers
pipeline_tag: image-text-to-text

base_model:
- liuhaotian/llava-v1.5-7b
base_model_relation: quantized

datasets:
- Lin-Chen/ShareGPT4V

metrics:
- accuracy

tags:
- llava
- llava-1.5
- vicuna
- vision-language-model
- multimodal
- quantization-aware-training
- qat
- awq
- int4
- w4g128
- grace
- knowledge-distillation
- efficient-vlm
- llava-eval
- icml-2026
---

# LLaVA-1.5-7B-GRACE-W4G128-AWQ

This repository contains the **real AWQ-packed INT4** deployment build of our
**GRACE-trained LLaVA-1.5-7B** checkpoint with **quantization-aware training (QAT)**
and **W4G128 group-wise INT4 quantization**. The weights here are stored as genuine
packed 4-bit integers (AutoAWQ GEMM layout: `qweight` / `qzeros` / `scales`), not as
fake-quantized BF16 tensors.

This model is associated with our ICML 2026 paper:

**[Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs](https://arxiv.org/abs/2601.22709)**  
Yanlong Chen, Amirhossein Habibian, Luca Benini, Yawei Li  
Accepted to the International Conference on Machine Learning (ICML 2026)

- **Paper:** https://arxiv.org/abs/2601.22709
- **DOI:** https://doi.org/10.48550/arXiv.2601.22709
- **Code:** https://github.com/ForeverBlue816/GRACE

---

## Model Details

- **Base model:** LLaVA-1.5-7B (Vicuna-7B-v1.5 + CLIP ViT-L/14-336)
- **Method:** GRACE: Gated Relational Alignment via Confidence-based Distillation
- **Quantization:** W4G128 group-wise INT4 QAT, packed to the AutoAWQ GEMM format
- **Training data:** ShareGPT4V
- **Evaluation setting:** LLaVA-style multimodal evaluation
- **Library:** Hugging Face Transformers (loaded through the GRACE / LLaVA-1.5 codebase)
- **Repository:** FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128-AWQ

---

## 📊 Results

This AWQ build is a **bit-exact repacking** of the LLaVA-1.5-7B GRACE W4G128 QAT
checkpoint: the integer weight codes are identical, and only the per-group scales
are stored in FP16. It therefore reproduces the **INT4 LLaVA-1.5 numbers reported
in the GRACE paper** rather than introducing a separate operating point. Please
refer to the paper for the full LLaVA-1.5 benchmark settings and results.

In practice the packing reduces the language-model weight footprint from
**≈14.2 GB (BF16) to ≈4.6 GB (≈3.1× smaller)**, with predictions matching the
QAT checkpoint up to FP16 scale rounding.

---

<a id="model-zoo"></a>
## 🤗 Model Zoo

| Model | Backbone | Bits | Group | Checkpoint description | HF Hub |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-GRACE-BF16 | Qwen3-VL-2B | bf16 | — | Full-precision GRACE checkpoint; used as the student initialization for the W8/W4 Qwen3-VL runs. | [FoeverBLUE/Qwen3-VL-2B-GRACE-BF16](https://huggingface.co/FoeverBLUE/Qwen3-VL-2B-GRACE-BF16) |
| Qwen3-VL-2B-GRACE-W8G128 | Qwen3-VL-2B | int8 | 128 | INT8 QAT checkpoint with group size 128; high-retention quantized Qwen3-VL student. | [FoeverBLUE/Qwen3-VL-2B-GRACE-W8G128](https://huggingface.co/FoeverBLUE/Qwen3-VL-2B-GRACE-W8G128) |
| Qwen3-VL-2B-GRACE-W4G128 | Qwen3-VL-2B | int4 | 128 | INT4 QAT checkpoint with group size 128; compact Qwen3-VL release retaining about 98% of the BF16 average. | [FoeverBLUE/Qwen3-VL-2B-GRACE-W4G128](https://huggingface.co/FoeverBLUE/Qwen3-VL-2B-GRACE-W4G128) |
| LLaVA-1.5-7B-GRACE-W4G128 | LLaVA-1.5-7B | int4 | 128 | INT4 QAT checkpoint from the GRACE paper with learned scales; fake-quantized BF16 weights plus a `qat_quantized_weights.bin` sidecar. | [FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128](https://huggingface.co/FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128) |
| **LLaVA-1.5-7B-GRACE-W4G128-AWQ** | LLaVA-1.5-7B | int4 | 128 | **This repo.** Real AWQ-packed (`qweight`/`qzeros`/`scales`) deployment build of the LLaVA-1.5 W4G128 checkpoint; loads through the GRACE codebase. | [FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128-AWQ](https://huggingface.co/FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128-AWQ) |

The `LLaVA-1.5-7B-GRACE-W4G128` repository is the QAT checkpoint (BF16 weights on
the INT4 grid plus a `qat_quantized_weights.bin` sidecar); **this** repository is
the same model packed into real 4-bit AWQ tensors for deployment.

---

## Intended Use

This model is intended for research on efficient vision-language models, quantization-aware training, knowledge distillation, and multimodal model compression.

Potential use cases include:

- Research on low-bit VLM deployment
- Analysis of QAT for multimodal large language models
- Efficient multimodal inference experiments
- Comparison with FP16, INT8, PTQ, AWQ, GPTQ, and other compression baselines

---

## Out-of-Scope Use

This model is not intended for safety-critical, medical, legal, financial, or high-stakes decision-making applications. The model may produce hallucinated, biased, or incorrect outputs and should be evaluated carefully before deployment.

---

## Training Data

The model was trained using ShareGPT4V-style multimodal instruction data. The training setup follows a LLaVA-style multimodal instruction-tuning/evaluation pipeline.

Dataset:

- `Lin-Chen/ShareGPT4V`

---

## Quantization Details

This checkpoint uses W4G128 group-wise INT4 quantization-aware training, packed
into the AutoAWQ GEMM format.

- **Weight precision:** INT4 (real packed `qweight` / `qzeros` / `scales`)
- **Grouping:** group size 128, group-wise along the input dimension
- **QAT scheme:** symmetric, signed, per-group Learned Step Size (LSQ) — codes in `[-8, 7]`, no zero point
- **AWQ mapping:** the symmetric QAT model maps onto AWQ's asymmetric GEMM format exactly by using a constant zero-point of `8` and `scales = exp(log_scale)`; the integer codes are bit-exact
- **Quantized modules:** the language-model linear layers only — `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj` across all decoder layers (224 layers for the 7B model)
- **Kept in FP16:** vision tower (CLIP), `mm_projector`, `embed_tokens`, `lm_head`, and all norms
- **Footprint:** ≈14.2 GB (BF16) → ≈4.6 GB (≈3.1× smaller)
- **Kernels:** optional fused INT4 kernels via `awq_ext` (`autoawq-kernels`); without them the loader falls back to a correct pure-PyTorch dequantization path

Because the language-model linears are stored as AWQ tensors rather than standard
`weight` matrices, this checkpoint requires the GRACE quantization-aware loading
code below — a plain `from_pretrained` will not reconstruct the INT4 layers.

---

## Files

- `config.json`: model configuration (`mm_vision_tower` should point to `openai/clip-vit-large-patch14-336`)
- `model-*.safetensors`: checkpoint shards (`qweight`/`qzeros`/`scales` for the quantized linears, FP16 for everything else)
- `model.safetensors.index.json`: checkpoint index file
- `awq_quantized_modules.json`: list of AWQ-packed module names + `bits` + `group_size` (required by the GRACE loader)
- `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`: tokenizer files
- `generation_config.json`: generation configuration

---

## Loading

This is the original LLaVA-1.5 architecture with AWQ-packed weights, so it loads
through the **GRACE / LLaVA-1.5 codebase** (a `transformers==4.37.2` LLaVA
environment), not through the standard `AutoModel` AWQ path.

```bash
# 1) Get the code + a LLaVA-1.5 environment (transformers==4.37.2, pinned)
git clone https://github.com/ForeverBlue816/GRACE && cd GRACE/deployment
python3 -m venv ~/llava && source ~/llava/bin/activate
pip install -U pip
pip install -r requirements.txt   # exact tested pins
pip install -e . --no-deps        # register the local `llava` package
# optional fused INT4 kernels:  pip install autoawq-kernels

# 2) Download this packed checkpoint
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128-AWQ"))
PY

# 3) One-shot inference on the bundled images/chinaairlines.jpg example
python scripts/deploy_awq_llava.py \
    --load-packed /path/to/downloaded/LLaVA-1.5-7B-GRACE-W4G128-AWQ \
    --image-file images/chinaairlines.jpg \
    --query "Please describe the scene in the picture in detail." \
    --conv-mode vicuna_v1 \
    --max-new-tokens 256
```

On the bundled `images/chinaairlines.jpg` this prints (≈5.9 GB peak GPU memory):

```text
=================== OUTPUT ===================
The image captures a moment at an airport, where a Boeing 787 Dreamliner, painted
in white and blue, is taxiing on the runway. The airplane is moving from the left
to the right of the frame, with its nose pointed towards the right side of the
image. The airplane is adorned with a pink flower on its tail, adding a touch of
color to the otherwise monochrome aircraft.

The background of the image provides a glimpse into the airport's infrastructure.
A control tower stands tall, overseeing the operations of the airport. A large
hangar is also visible, likely housing other aircraft or serving as a maintenance
facility.

The sky above is a clear blue, suggesting good weather conditions for flight. The
grass surrounding the runway is a vibrant green, indicating it might be spring or
summer. The overall scene is a typical day at an airport, with the Boeing 787
Dreamliner preparing for its next journey.
==============================================
```

Install `autoawq-kernels` for the fused INT4 kernels (large speedup); without them
the model still runs correctly via a slower pure-PyTorch dequantization path.

Or load it programmatically with the GRACE helpers:

```python
import os, glob, json
from safetensors.torch import load_file
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.quantize import build_awq_skeleton

d = "/path/to/downloaded/LLaVA-1.5-7B-GRACE-W4G128-AWQ"
meta = json.load(open(os.path.join(d, "awq_quantized_modules.json")))

tokenizer, model, image_processor, _ = load_pretrained_model(
    d, None, get_model_name_from_path(d), device_map="cuda", device="cuda")

# replace the language-model linears with AWQ modules, then load packed weights
build_awq_skeleton(model, meta["modules"], bits=meta["bits"],
                   group_size=meta["group_size"], device="cuda")
sd = {}
for f in glob.glob(os.path.join(d, "*.safetensors")):
    sd.update(load_file(f))
prefixes = tuple(n + "." for n in meta["modules"])
model.load_state_dict({k: v for k, v in sd.items() if k.startswith(prefixes)}, strict=False)
model.eval()
```

The CLIP vision tower is loaded from `openai/clip-vit-large-patch14-336` via the
`mm_vision_tower` field; install `autoawq-kernels` for fused INT4 kernels (optional —
the loader otherwise uses a correct PyTorch dequantization path).

---

## Evaluation

The checkpoint follows a **LLaVA-style multimodal evaluation protocol** and is
evaluated with greedy decoding. Representative benchmarks include:

- VQAv2, GQA, TextVQA, POPE, MME
- ScienceQA, SEED-Bench, MMBench

Please refer to the associated GRACE paper for detailed evaluation settings and
results, and to the GRACE repository for the evaluation scripts.

---

## Important Notes

- This repository is the **real packed INT4** form of the model (not fake-quantized BF16). The language-model linears are stored as AWQ `qweight`/`qzeros`/`scales`.
- A plain `from_pretrained` call will **not** reconstruct the INT4 layers; use the GRACE loader shown above (`build_awq_skeleton` + `load_state_dict`, or `scripts/deploy_awq_llava.py --load-packed`).
- `config.json`'s `mm_vision_tower` must resolve to `openai/clip-vit-large-patch14-336` (or a valid local CLIP path) for the vision tower to load.
- Specialized kernels (`autoawq-kernels`) are required to realize practical INT4 speed; without them the model still runs correctly and at reduced storage via a slower dequantization path.

---

## Limitations

- This model is released for research purposes.
- The packed checkpoint requires the GRACE quantization-aware loading code; it is not a drop-in standard Transformers AWQ checkpoint.
- Performance may vary depending on the evaluation codebase, preprocessing, generation parameters, and multimodal benchmark implementation.
- Users should follow the license and usage restrictions of the original LLaVA-1.5 / Vicuna base model and the training data.

---

## Citation

If you use this model, please cite the corresponding GRACE work:

```bibtex
@article{chen2026gated,
  title={Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs},
  author={Chen, Yanlong and Habibian, Amirhossein and Benini, Luca and Li, Yawei},
  journal={arXiv preprint arXiv:2601.22709},
  year={2026}
}
```

Please also cite the original LLaVA and Vicuna work when using this model.

---

## License

This model is released under the Apache-2.0 license unless otherwise specified. Users should also comply with the license and usage terms of the base model and training data.
