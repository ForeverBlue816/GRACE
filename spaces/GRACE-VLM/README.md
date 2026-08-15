---
title: GRACE-VLM
emoji: 🦢
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.10"
sdk_version: 5.49.1
app_file: app.py
pinned: true
short_description: Try GRACE-VLM on an image, then deploy the INT4 build.
suggested_hardware: zero-a10g
startup_duration_timeout: 1h
preload_from_hub:
  - ForeverBlue/Qwen3-VL-2B-GRACE-BF16
models:
  - ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ
  - ForeverBlue/Qwen3-VL-2B-GRACE-BF16
tags:
  - vision-language-model
  - multimodal
  - int4
  - awq
  - knowledge-distillation
  - arxiv:2601.22709
---

# GRACE-VLM

Live demo for **GRACE-VLM: INT4 Quantization-Aware Distillation for
Vision-Language Models**, accepted at ICML 2026. Read the paper at
[arXiv:2601.22709](https://arxiv.org/abs/2601.22709).

Upload an image and ask a question to run the GRACE 2B BF16 checkpoint on free
ZeroGPU hardware. For genuine packed INT4 inference, use
[`ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ`](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ)
with the copy-ready [GRACE loader](https://github.com/ForeverBlue816/GRACE#quick-start-real-int4).
