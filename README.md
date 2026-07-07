<p align="center">
  <img src="assets/swan.png" alt="GRACE" width="200"/>
</p>

<h1 align="center">GRACE</h1>

<p align="center"><b>Gated Relational Alignment via Confidence-based Distillation<br/>for Quantization-Aware Training of Vision-Language Models</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.22709"><img src="https://img.shields.io/badge/ICML%202026-Accepted-1f6feb?style=flat-square" alt="ICML 2026"/></a>
  <a href="https://arxiv.org/abs/2601.22709"><img src="https://img.shields.io/badge/arXiv-2601.22709-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"/></a>
  <a href="https://huggingface.co/ForeverBlue"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffce1c?style=flat-square" alt="Hugging Face Models"/></a>
  <img src="https://img.shields.io/badge/python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.5+"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-4caf50?style=flat-square" alt="License: Apache 2.0"/></a>
</p>

<p align="center"><b>Official PyTorch implementation of GRACE, accepted at ICML 2026.</b></p>

<p align="center">
  📄 <a href="https://arxiv.org/abs/2601.22709"><b>Paper</b></a>
  &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/ForeverBlue"><b>Models</b></a>
  &nbsp;|&nbsp;
  📦 <a href="https://huggingface.co/datasets/Lin-Chen/ShareGPT4V"><b>Training Data</b></a>
</p>

<details>
<summary>📖 <b>Table of Contents</b></summary>

- [📊 Results](#results)
- [🤗 Model Zoo](#model-zoo)
- [📁 Repository Layout](#repository-layout)
- [⚙️ Environment Setup](#environment-setup)
- [🗂️ Data Preparation](#data-preparation)
- [🚀 Training](#training)
- [📈 Evaluation](#evaluation)
- [📦 Deployment (INT4 / AWQ Inference)](#deployment)
- [📝 Citation](#citation)
- [🙏 Acknowledgements](#acknowledgements)
- [⭐ Star History](#star-history)
- [📜 License](#license)

</details>

GRACE is a quantization-aware training (QAT) framework for vision-language
models. The goal is simple to state: push a small student VLM down to 4-bit
weights and still keep almost all of the accuracy you would have had at full
precision. The trick is that GRACE trains quantization and distillation
*together*, instead of quantizing a model after the fact, so the student learns
to be a good low-bit model from the start.

Three pieces make this work:

- **GDKD:** confidence-gated *Decoupled Knowledge Distillation* (TCKD + NCKD),
  where the trade-off `β` is adapted online by an Information-Bottleneck
  controller.
- **RCKA:** *Relational Centered Kernel Alignment* on penultimate-layer visual
  tokens, which aligns the student's relational geometry to the teacher's.
- **Group-wise LSQ QAT:** learned per-group weight scales (W4 / W8, group size
  128) on the LLM and the MLP projector, with the ViT kept frozen.

Training optimizes

```
L_total = L_CE  +  β · L_GDKD  +  ω · L_RCKA
```

where `β` is driven by the IB controller (`τ`, `η`) and `ω` is warmed up
linearly. The defaults follow the values reported in the paper (Table 6); see
[finetune_qwen3vl_2b_grace.slurm](qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace.slurm)
for the full hyper-parameter list.

The reference implementation in this repo applies GRACE to
**Qwen3-VL-2B-Instruct** (student) distilled from **Qwen3-VL-8B-Instruct**
(teacher).

<p align="center">
  <img src="assets/grace_architecture.png" alt="GRACE architecture" width="100%"/>
</p>

---

<a id="results"></a>
## 📊 Results

Comparison on 7 VLM benchmarks. The 8B model is the distillation **teacher**
(reference upper bound); every GRACE-Qwen3 variant is a **2B** student. The best
result among the 2B Qwen3-VL models is in **bold**.

We release GRACE on Qwen3-VL here because it is the most current backbone and
gives a fairer, up-to-date point of comparison, with the vanilla
Qwen3-VL-2B-Instruct as the baseline. The paper itself reports GRACE on
LLaVA-1.5 and Qwen2-VL, and we additionally release the LLaVA-1.5 W4G128 (INT4)
checkpoint from the paper in the model zoo below.

| Model | Params | Precision | HallB | MMBench | ScienceQA | AI2D | MMMU | SEED | MMStar | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-8B *(teacher, ref.)* | 8B | BF16 | 61.1 | 84.5 | 85.0 | 85.7 | 69.6 | 77.5 | 70.9 | 76.3 |
| Qwen3-VL-2B *(baseline)*      | 2B | BF16 | 51.4 | 78.4 | 81.4 | 76.9 | 53.4 | 71.2 | 58.3 | 67.3 |
| **Qwen3-VL-2B-GRACE**         | 2B | BF16 | **66.9** | **86.4** | **86.2** | **81.3** | **72.1** | **76.7** | **67.3** | **76.7** |
| Qwen3-VL-2B-GRACE (W8G128)    | 2B | INT8 | 66.1 | 85.5 | 85.3 | 80.4 | 71.3 | 75.9 | 66.5 | 75.9 |
| Qwen3-VL-2B-GRACE (W4G128)    | 2B | INT4 | 65.4 | 84.6 | 84.3 | 79.5 | 70.5 | 75.1 | 65.8 | 75.0 |

> GRACE lifts the Qwen3-VL-2B baseline by **+9.4 avg** and matches or slightly
> exceeds the 8B teacher on average (76.7 vs. 76.3) at roughly a quarter of the
> parameters. The W4G128 (INT4) model still retains **98%** of the BF16 average.

---

<a id="model-zoo"></a>
## 🤗 Model Zoo

| Model | Backbone | Bits | Group | Checkpoint description | HF Hub |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-GRACE-BF16 | Qwen3-VL-2B | bf16 | n/a | Full-precision GRACE checkpoint; used as the student initialization for the W8/W4 Qwen3-VL runs. | [ForeverBlue/Qwen3-VL-2B-GRACE-BF16](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-BF16) |
| Qwen3-VL-2B-GRACE-W8G128 | Qwen3-VL-2B | int8 | 128 | INT8 QAT checkpoint with group size 128; high-retention quantized Qwen3-VL student. | [ForeverBlue/Qwen3-VL-2B-GRACE-W8G128](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W8G128) |
| Qwen3-VL-2B-GRACE-W4G128 | Qwen3-VL-2B | int4 | 128 | INT4 QAT checkpoint with group size 128; compact Qwen3-VL release retaining about 98% of the BF16 average. | [ForeverBlue/Qwen3-VL-2B-GRACE-W4G128](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128) |
| Qwen3-VL-2B-GRACE-W4G128-AWQ | Qwen3-VL-2B | int4 | 128 | **Real AWQ-packed** (`qweight`/`qzeros`/`scales`) deployment build of the W4G128 student; drop-in INT4 inference via the GRACE deploy loader. | [ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ) |
| LLaVA-1.5-7B-GRACE-W4G128 | LLaVA-1.5-7B | int4 | 128 | INT4 QAT checkpoint from the GRACE paper with learned scales; released for reproducing the LLaVA-1.5 experiments. | [ForeverBlue/LLaVA-1.5-7B-GRACE-W4G128](https://huggingface.co/ForeverBlue/LLaVA-1.5-7B-GRACE-W4G128) |
| LLaVA-1.5-7B-GRACE-W4G128-AWQ | LLaVA-1.5-7B | int4 | 128 | Real AWQ-packed deployment build of the LLaVA-1.5 W4G128 checkpoint; loads through the GRACE / LLaVA-1.5 codebase. | [ForeverBlue/LLaVA-1.5-7B-GRACE-W4G128-AWQ](https://huggingface.co/ForeverBlue/LLaVA-1.5-7B-GRACE-W4G128-AWQ) |

The BF16 Qwen3-VL checkpoint is the full-precision GRACE student that we use to
initialize the W8 and W4 Qwen3-VL runs. The LLaVA-1.5 W4G128 checkpoint
corresponds to the paper setting and ships with the GRACE-specific QAT
quantized weights, so you can reproduce the INT4 LLaVA experiments directly. The
two `*-AWQ` repositories are the real 4-bit packed builds for deployment; see
[Deployment](#deployment) for how to run them or repack a checkpoint yourself.

**Quick load:**

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ckpt = "ForeverBlue/Qwen3-VL-2B-GRACE-W4G128"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    ckpt, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(ckpt)
```

---

<a id="repository-layout"></a>
## 📁 Repository Layout

```
.
├── qwen-vl-finetune/        # Training + INT4 deployment for Qwen3-VL
│   ├── qwenvl/
│   │   ├── data/            # Dataset registry + LLaVA-style loader
│   │   ├── quantize/        # QAT→AWQ packing + WQLinear_GEMM (Qwen3-VL)
│   │   └── train/
│   │       ├── train_qwen.py        # plain BF16 SFT
│   │       ├── train_qwen_qat.py    # group-wise LSQ QAT
│   │       ├── train_qwen_grace.py  # GRACE = QAT + GDKD + RCKA
│   │       ├── qat_modules.py       # LSQ fake-quant + save hooks
│   │       └── grace_modules.py     # GDKD, RCKA, IB controller
│   └── scripts/
│       ├── finetune_qwen3vl_2b_bf16.slurm   # BF16 SFT (baseline)
│       ├── finetune_qwen3vl_2b_sft.slurm    # BF16 SFT (alt config)
│       ├── finetune_qwen3vl_2b_qat.slurm    # QAT only (ablation)
│       ├── finetune_qwen3vl_2b_grace.slurm  # GRACE
│       └── deploy_awq_qwen.py               # pack & run real INT4 inference
├── evaluation/              # lmms-eval driver + per-benchmark configs
├── deployment/              # LLaVA-1.5 tree: QAT training + AWQ INT4 packing/inference
│   ├── llava/quantize/      # QAT to AWQ conversion + WQLinear_GEMM kernels
│   ├── scripts/deploy_awq_llava.py            # pack & run real INT4 inference
│   └── scripts/v1_5/finetune_qat.{sh,slurm}   # LLaVA-1.5 QAT launchers
├── qwen-vl-utils/           # Qwen3-VL multi-modal preprocessing helpers
├── cookbooks/               # Qwen3-VL inference / capability demos
├── docker/                  # CUDA 12.8 image for web demo
├── web_demo_mm.py           # Multi-modal Gradio demo
├── assets/                  # README figures (architecture, icon)
└── requirements.txt         # Pinned versions for the qwen3vl venv
```

---

<a id="environment-setup"></a>
## ⚙️ Environment Setup

GRACE was trained on 16 × A100 (64 GB) GPUs across 4 nodes. The reference SLURM
scripts pin the host toolchain in their `module load` block:

| Component | Version |
| --- | --- |
| CUDA driver / runtime (host) | **12.3** |
| GCC                          | **12.2.0** |
| Python                       | **3.11** |
| PyTorch                      | **2.5.1** (cu121 wheels, forward-compatible with the CUDA 12.3 driver) |
| flash-attn                   | **2.7.2.post1** |
| DeepSpeed                    | **0.15.4** (ZeRO-2) |
| transformers                 | **5.9.0** |
| accelerate                   | **1.13.0** |

A frozen export of the full virtual environment is in
[requirements.txt](requirements.txt).

### 1. Clone the repository

```bash
git clone https://github.com/ForeverBlue816/GRACE.git
cd GRACE
```

### 2. Build the venv from scratch

```bash
# (a) System modules. This is the Leonardo example; adapt it to your cluster.
module purge
module load profile/deeplrn
module load cuda/12.3
module load gcc/12.2.0

# (b) Create and activate the venv
python3.11 -m venv "${HOME}/qwen3vl"
source "${HOME}/qwen3vl/bin/activate"
pip install -U pip wheel setuptools

# (c) PyTorch + CUDA runtime (cu121 wheels)
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# (d) Everything else, pinned to the released training env
pip install -r requirements.txt

# (e) flash-attn. Build this AFTER torch is installed, or the build will fail.
pip install flash-attn==2.7.2.post1 --no-build-isolation

# (f) Local utility package (image / video preprocessing for Qwen3-VL)
pip install -e qwen-vl-utils/
```

### 3. Pre-stage the model weights

Download the teacher and student weights on the login node (or any
internet-reachable host) before launching a compute job:

```bash
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
    --local-dir "${SCRATCH_ROOT}/Qwen3-VL-2B-Instruct"
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir "${SCRATCH_ROOT}/Qwen3-VL-8B-Instruct"
```

---

<a id="data-preparation"></a>
## 🗂️ Data Preparation

GRACE is trained on the two ShareGPT4V annotation files (LLaVA-style schema,
`image` + `conversations[from/value]`):

| # | Annotation JSON | Size | Hugging Face |
| - | --- | ---: | --- |
| 1 | `sharegpt4v_instruct_gpt4-vision_cap100k.json` | 134 MB | [Lin-Chen/ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_instruct_gpt4-vision_cap100k.json) |
| 2 | `sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json` | 1.2 GB | [Lin-Chen/ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json) |

`sharegpt4v_mix665k_*` is the main SFT mix used in the paper, while
`sharegpt4v_instruct_gpt4-vision_cap100k.json` is the original GPT-4V caption
set.

### 1. Download the annotation JSONs

```bash
export SHAREGPT4V_ROOT=/path/to/ShareGPT4V
mkdir -p "${SHAREGPT4V_ROOT}"

huggingface-cli download Lin-Chen/ShareGPT4V \
    sharegpt4v_instruct_gpt4-vision_cap100k.json \
    sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
    --repo-type dataset \
    --local-dir "${SHAREGPT4V_ROOT}"
```

### 2. Download the image archives

The ShareGPT4V annotations point at images under `${SHAREGPT4V_ROOT}/data/`.
Download and unpack the sources below. Only the ones the JSONs actually
reference are required.

| Source | URL |
| --- | --- |
| LAION-CC-SBU-558K | [images.zip](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip) |
| COCO              | [train2017.zip](http://images.cocodataset.org/zips/train2017.zip) |
| SAM (subset)      | [segment-anything-downloads](https://ai.meta.com/datasets/segment-anything-downloads/), files `000000~000050.tar`. For SFT-only you can use the 9k subset [here](https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link). |
| GQA               | [images.zip](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) |
| OCR-VQA           | [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), saving all images as `.jpg`. |
| TextVQA           | [train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) |
| Visual Genome     | [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) |
| WebData (academic use only) | [drive folder](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing) |

### 3. Final directory layout

```
${SHAREGPT4V_ROOT}/
├── sharegpt4v_instruct_gpt4-vision_cap100k.json
├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
└── data/
    ├── llava/llava_pretrain/images/
    ├── coco/train2017/
    ├── sam/images/
    ├── gqa/images/
    ├── ocr_vqa/images/
    ├── textvqa/train_images/
    ├── vg/VG_100K/
    ├── vg/VG_100K_2/
    ├── share_textvqa/images/
    ├── web-celebrity/images/
    ├── web-landmark/images/
    └── wikiart/images/
```

The dataset registry that resolves these paths lives in
[qwen-vl-finetune/qwenvl/data/__init__.py](qwen-vl-finetune/qwenvl/data/__init__.py),
which reads `SHAREGPT4V_ROOT` from the environment.

---

<a id="training"></a>
## 🚀 Training

All three recipes are launched with `sbatch`. The full reference run uses
16 × A100 (64 GB) across 4 nodes, DeepSpeed ZeRO-2, BF16, and an effective batch
size of 512. Adjust `--nodes`, `PER_DEVICE_BATCH`, and `GRAD_ACCUM` to fit your
own cluster.

### 1. BF16 SFT baseline

Optional, and also the source of our released `*-BF16` checkpoint:

```bash
sbatch qwen-vl-finetune/scripts/finetune_qwen3vl_2b_bf16.slurm
```

### 2. QAT-only baseline

This is the ablation of GRACE without distillation:

```bash
# W4 G128 (default)
sbatch qwen-vl-finetune/scripts/finetune_qwen3vl_2b_qat.slurm

# W8 G128
sbatch --export=ALL,QAT_BITS=8 qwen-vl-finetune/scripts/finetune_qwen3vl_2b_qat.slurm
```

### 3. GRACE (full method)

```bash
# W4 G128: produces ForeverBlue/Qwen3-VL-2B-GRACE-W4G128
sbatch qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace.slurm

# W8 G128: produces ForeverBlue/Qwen3-VL-2B-GRACE-W8G128
sbatch --export=ALL,QAT_BITS=8 qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace.slurm
```

Common environment-variable overrides (apply to all scripts):

| Variable | Default | Description |
| --- | --- | --- |
| `SHAREGPT4V_ROOT` | `PATH_TO_SHAREGPT4V_ROOT` | Root of the ShareGPT4V tree. |
| `DATASETS` | `sharegpt4v_mix665k` | Comma-separated; a `%NN` suffix downsamples. |
| `MODEL_NAME_OR_PATH` | `${SCRATCH_ROOT}/Qwen3-VL-2B-Instruct` | Student initialization. |
| `TEACHER_MODEL_PATH` | `${SCRATCH_ROOT}/Qwen3-VL-8B-Instruct` | GRACE only. |
| `QAT_BITS` / `QAT_GROUP_SIZE` | `4` / `128` | LSQ fake-quant config. |
| `OUTPUT_DIR` | `${CKPT_ROOT}/${RUN_NAME}` | Auto-resumes from the latest `checkpoint-*`. |

GRACE-specific knobs (defaults follow the paper):

| Variable | Default | Meaning |
| --- | --- | --- |
| `DKD_TEMPERATURE` | `2.0` | KD temperature `T`. |
| `DKD_ALPHA` / `DKD_BETA` | `1.0` / `4.0` | TCKD / NCKD weights. |
| `RCKA_WEIGHT` | `3.0` | `ω` for L_RCKA. |
| `RCKA_LAYER` | `-2` | Hidden-state index used by RCKA. |
| `IB_TAU` / `IB_ETA` | `3.0` / `0.003` | IB controller target / step size. |
| `IB_BETA_INIT/MIN/MAX` | `0.5` / `0.1` / `1.0` | `β` schedule bounds. |
| `RCKA_WARMUP_STEPS` | `400` | Linear warmup for RCKA. |

---

<a id="evaluation"></a>
## 📈 Evaluation

We score every checkpoint with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
The per-benchmark configs live under
[evaluation/](https://github.com/ForeverBlue816/GRACE/tree/main/evaluation).

```bash
# Example: evaluate the released W4 checkpoint on ScienceQA
sbatch --export=ALL,MODEL=ForeverBlue/Qwen3-VL-2B-GRACE-W4G128 \
       evaluation/ScienceQA/eval_scienceqa.slurm
```

---

<a id="deployment"></a>
## 📦 Deployment — INT4 / AWQ Inference

Both the Qwen3-VL and LLaVA-1.5 GRACE checkpoints ship as deployable **real
INT4** builds. A GRACE QAT checkpoint stores BF16 weights projected onto the
INT4 grid, together with a `qat_quantized_weights.bin` sidecar that holds the
learned per-group scales. For deployment we **pack** the quantized
language-model layers into genuine 4-bit AutoAWQ tensors (`qweight`, `qzeros`,
`scales`) that are compatible with AWQ-style INT4 GEMM kernels.

This packing step is **bit-exact** with respect to the learned integer weight
codes: the INT4 codes are unchanged, and only the per-group scales are stored in
FP16. The two backbones share the same conversion math but use different
runtimes:

- **Qwen3-VL** packs through
  [`qwen-vl-finetune/qwenvl/quantize/`](qwen-vl-finetune/qwenvl/quantize) and runs
  in the **`qwen3vl` training environment** (transformers 5.9).
- **LLaVA-1.5** packs through
  [`deployment/llava/quantize/`](deployment/llava/quantize) and runs in a
  separate **`transformers==4.37.2`** environment.

Installing the fused INT4 kernels (`pip install autoawq-kernels`) in either
environment enables a speedup; without them the model still runs through a
correct, slower pure-PyTorch dequantization path.

<details>
<summary><b>How the conversion works (symmetric LSQ-QAT → asymmetric AWQ)</b></summary>

GRACE QAT is **symmetric signed** per group: code `q ∈ [-8, 7]`, a per-group
scale `s`, **no zero point**, so the dequantized weight is `W = q · s` (groups of
`group_size = 128` along the input dim). AWQ's GEMM kernel is **asymmetric
unsigned**: `W = scales · (q_awq − zeros)` with `q_awq ∈ [0, 15]`. The two line
up *exactly* with a constant zero-point:

```
zeros  = 8  (= 2^(bits-1))
scales = s  (= exp(log_scale), stored FP16)
q_awq  = q + 8 ∈ [0, 15]
⇒  scales · (q_awq − 8) = s · q = W      (no error beyond FP16 scale rounding)
```

Only the per-group scales change dtype; the integer codes are identical. The
converters
([qwenvl/quantize/qat_to_awq.py](qwen-vl-finetune/qwenvl/quantize/qat_to_awq.py)
for Qwen3-VL,
[llava/quantize/qat_to_awq.py](deployment/llava/quantize/qat_to_awq.py) for
LLaVA) treat the sidecar as the source of truth for which layers were quantized,
pack each one, and (unless `--no-verify`) assert the max int-code mismatch is
`0`. Only the language-model linears (`self_attn.{q,k,v,o}_proj` and
`mlp.{gate,up,down}_proj` across all decoder layers) are quantized; the vision
tower, projector / merger, embeddings, `lm_head`, and norms stay in their
original dtype.

</details>

### Qwen3-VL-2B (AWQ INT4)

| Repository | Stored format | Recommended use |
| --- | --- | --- |
| [Qwen3-VL-2B-GRACE-W4G128](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128) | BF16 weights on the INT4 grid, plus the `qat_quantized_weights.bin` sidecar | research / re-packing; the source for the conversion below |
| [Qwen3-VL-2B-GRACE-W4G128-AWQ](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ) | real packed `qweight` / `qzeros` / `scales` | drop-in INT4 inference |

The Qwen3-VL deployment uses the **same `qwen3vl` training environment** (no
extra environment needed; `qwen_vl_utils` is already installed by the setup
above). Run the commands from the `qwen-vl-finetune/` directory.

**Run the released AWQ model:**

```bash
cd qwen-vl-finetune

# Download the packed checkpoint and print its local path
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ"))
PY

# Run inference (substitute the path printed above for /path/to/...)
python scripts/deploy_awq_qwen.py \
    --load-packed /path/to/Qwen3-VL-2B-GRACE-W4G128-AWQ \
    --image ../deployment/images/chinaairlines.jpg \
    --query "Describe this image in detail." \
    --max-new-tokens 256
```

**Or convert the BF16 QAT checkpoint → AWQ yourself:**

```bash
# Load → pack to real INT4 → run → persist the packed model
python scripts/deploy_awq_qwen.py \
    --model-path /path/to/Qwen3-VL-2B-GRACE-W4G128 \
    --image ../deployment/images/chinaairlines.jpg \
    --query "Describe this image in detail." \
    --max-new-tokens 256 \
    --save-dir ./qwen3vl-2b-w4-awq-packed
```

This loads the checkpoint, reads its `qat_quantized_weights.bin` sidecar, swaps
every quantized language-model linear (`language_model.layers.*`
`self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj`) for an AWQ
`WQLinear_GEMM`, verifies the packing is bit-exact (`max int-code error = 0`),
runs inference, and writes the packed model plus `awq_quantized_modules.json` to
`--save-dir`. Reload it instantly with `--load-packed ./qwen3vl-2b-w4-awq-packed`.
The vision tower, merger, embeddings, `lm_head`, and norms stay BF16. For a fast
LLM-only smoke test, drop `--image` and add `--text-only`.

Example image:

<p align="center">
  <img src="deployment/images/chinaairlines.jpg" alt="chinaairlines.jpg example" width="55%"/>
</p>

Example output from the **Qwen3-VL-2B-GRACE-W4G128-AWQ** model:

```text
=================== OUTPUT ===================
The image captures a moment on an airport runway, where a China Airlines Boeing
777-300ER airplane is in the process of taxiing. The airplane, painted in a
striking combination of white and blue, is adorned with a pink flower design on
its tail, adding a touch of elegance to its appearance. The words "China Airlines"
and "Boeing" are prominently displayed on the side of the airplane, indicating its
affiliation and model.

The airplane is moving towards the right side of the image, suggesting it's either
preparing for takeoff or has just landed. The runway beneath it is a testament to
human engineering, with its smooth surface designed for the safe and efficient
movement of aircraft.

In the background, the airport's infrastructure is visible, including buildings and
other airport facilities. These structures, while not the main focus of the image,
provide context to the setting and the purpose of the airplane's journey.

Overall, the image presents a snapshot of modern aviation, showcasing the marvels
of technology and design that enable air travel.
==============================================
```

> The INT4 GRACE student correctly reads the livery ("China Airlines"), identifies
> the Boeing 777-300ER, and grounds the description in the runway scene — at
> roughly a quarter of the BF16 weight footprint.

### LLaVA-1.5-7B (AWQ INT4)

| Repository | Stored format | Recommended use |
| --- | --- | --- |
| [LLaVA-1.5-7B-GRACE-W4G128](https://huggingface.co/ForeverBlue/LLaVA-1.5-7B-GRACE-W4G128) | BF16 weights projected onto the INT4 grid, plus the `qat_quantized_weights.bin` sidecar | research, inspection, and re-packing experiments |
| [LLaVA-1.5-7B-GRACE-W4G128-AWQ](https://huggingface.co/ForeverBlue/LLaVA-1.5-7B-GRACE-W4G128-AWQ) | real packed AWQ tensors: `qweight`, `qzeros`, `scales` | ready-to-run INT4 inference through the GRACE loader |

For the 7B LLaVA-1.5 model the packing reduces the language-model weight
footprint from **≈14.2 GB (BF16) to ≈4.6 GB**, an approximately **3.1×**
reduction. The LLaVA-1.5 deployment code lives under
[`deployment/`](deployment): a vendored LLaVA-1.5 codebase extended with the
GRACE QAT and AWQ loading utilities. It needs a dedicated
`transformers==4.37.2` environment, separate from the `qwen3vl` training
environment. **Run all commands in this section from the `deployment/`
directory.**

#### 1. Environment

The exact tested package versions are in
[`deployment/requirements.txt`](deployment/requirements.txt) for reproducibility.

```bash
cd deployment

# Create a fresh venv for the LLaVA-1.5 deployment stack.
# Do not reuse the qwen3vl training environment.
python3 -m venv ~/llava
source ~/llava/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e . --no-deps

# Optional acceleration packages. Install these after PyTorch is available.
pip install flash-attn==2.5.8 --no-build-isolation
pip install autoawq-kernels
```

> Tested on an NVIDIA A100 with a CUDA 12.x driver using the versions in
> `requirements.txt`. The pinned `torch==2.1.2+cu121` package ships its own CUDA
> runtime, so a system CUDA toolkit and compatible compiler are only needed when
> building packages such as `flash-attn` or `autoawq-kernels`.

#### 2. Run the released AWQ model

Download the packed INT4 checkpoint and run one-shot inference on the bundled
[`chinaairlines.jpg`](deployment/images/chinaairlines.jpg) example:

```bash
# Download the packed checkpoint and print its local path
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("ForeverBlue/LLaVA-1.5-7B-GRACE-W4G128-AWQ"))
PY

# Run inference (substitute the path printed above for /path/to/...)
python scripts/deploy_awq_llava.py \
    --load-packed /path/to/LLaVA-1.5-7B-GRACE-W4G128-AWQ \
    --image-file images/chinaairlines.jpg \
    --query "Please describe the scene in the picture in detail." \
    --conv-mode vicuna_v1 \
    --max-new-tokens 256
```

Example output:

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
[deploy] generated in 59.56s (4.3 tok/s nominal)
[deploy] peak GPU mem during generate: 5.91 GB
```

Here the packed INT4 checkpoint runs with a peak GPU memory footprint of about
**5.9 GB**. The **4.3 tokens/s** figure is the pure-PyTorch dequantization
fallback; installing `autoawq-kernels` switches on the fused INT4 kernels and
substantially improves throughput.

`--load-packed` reuses the standard LLaVA architecture, tokenizer, and CLIP
vision tower, then reconstructs the AWQ-packed modules listed in
`awq_quantized_modules.json` and loads the packed INT4 tensors directly. No
re-packing happens at inference time. The CLIP vision tower is resolved from the
`mm_vision_tower` field in the model config; it defaults to
`openai/clip-vit-large-patch14-336`, so the host needs either internet access on
the first run or a valid local CLIP path.

> **Note:** this checkpoint follows the original LLaVA-1.5 architecture with
> AWQ-packed weights, so it must be loaded through the GRACE deployment code. A
> plain `from_pretrained` call will not reconstruct the INT4 layers.

#### 3. Convert a BF16 QAT checkpoint → AWQ yourself

To rebuild the AWQ checkpoint from the released QAT checkpoint (or to pack one
you trained), load the BF16 fake-quant checkpoint, pack it into real 4-bit, and
persist it with `--save-dir`:

```bash
# Download the QAT (fake-quant BF16 + sidecar) checkpoint
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("ForeverBlue/LLaVA-1.5-7B-GRACE-W4G128"))
PY

# Load fp16 → pack to real INT4 → run → persist the packed model
python scripts/deploy_awq_llava.py \
    --model-path /path/to/LLaVA-1.5-7B-GRACE-W4G128 \
    --image-file images/chinaairlines.jpg \
    --query "Please describe the scene in the picture in detail." \
    --conv-mode vicuna_v1 \
    --max-new-tokens 256 \
    --save-dir ./checkpoints/llava-w4-awq-packed
```

This loads the BF16 checkpoint, reads its `qat_quantized_weights.bin` sidecar,
swaps every quantized LLM linear for an AWQ `WQLinear_GEMM`, verifies the packing
is bit-exact, runs inference, and writes the packed model (plus
`awq_quantized_modules.json`) to `--save-dir`. After that you can reload it
instantly with `--load-packed ./checkpoints/llava-w4-awq-packed`. The
`--save-dir` of one run is exactly the `--load-packed` of the next. For a fast
LLM-only smoke test, drop `--image-file` and add `--text-only`.

#### 4. Load it programmatically

```python
import os, glob, json
from safetensors.torch import load_file
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.quantize import build_awq_skeleton

d = "/path/to/LLaVA-1.5-7B-GRACE-W4G128-AWQ"
meta = json.load(open(os.path.join(d, "awq_quantized_modules.json")))

tokenizer, model, image_processor, _ = load_pretrained_model(
    d, None, get_model_name_from_path(d), device_map="cuda", device="cuda"
)

# Replace the LLM linears with AWQ modules, then load the packed weights
build_awq_skeleton(
    model, meta["modules"], bits=meta["bits"],
    group_size=meta["group_size"], device="cuda"
)
sd = {}
for f in glob.glob(os.path.join(d, "*.safetensors")):
    sd.update(load_file(f))
prefixes = tuple(n + "." for n in meta["modules"])
model.load_state_dict(
    {k: v for k, v in sd.items() if k.startswith(prefixes)}, strict=False
)
model.eval()
```

---

<a id="citation"></a>
## 📝 Citation

If GRACE or the released checkpoints help your research, please cite:

```bibtex
@inproceedings{chen2026gated,
  title     = {Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs},
  author    = {Chen, Yanlong and Habibian, Amirhossein and Benini, Luca and Li, Yawei},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.22709}
}
```

---

<a id="acknowledgements"></a>
## 🙏 Acknowledgements

GRACE builds on the public Qwen3-VL release and the
[Qwen2.5-VL fine-tuning code](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune).
The ShareGPT4V training data comes from
[Lin-Chen/ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V), and
evaluation is powered by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
A big thank-you to all of these communities.

**Questions or contributions?** Open an
[issue](https://github.com/ForeverBlue816/GRACE/issues) or a pull request, or
reach out to Yanlong Chen at
[yanlchen@student.ethz.ch](mailto:yanlchen@student.ethz.ch).

---

<a id="star-history"></a>
## Star History

<a href="https://www.star-history.com/?type=date&repos=ForeverBlue816%2FGRACE">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ForeverBlue816/GRACE&type=date&theme=dark&legend=top-left&sealed_token=BeXsnKcWfn3spTqfDcb9eRNe3l99jQkm6a9vC_CHq0Ikj_-o8ZgEF_5aEF2MX4vckINFfiU" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ForeverBlue816/GRACE&type=date&legend=top-left&sealed_token=BeXsnKcWfn3spTqfDcb9eRNe3l99jQkm6a9vC_CHq0Ikj_-o8ZgEF_5aEF2MX4vckINFfiU" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ForeverBlue816/GRACE&type=date&legend=top-left&sealed_token=BeXsnKcWfn3spTqfDcb9eRNe3l99jQkm6a9vC_CHq0Ikj_-o8ZgEF_5aEF2MX4vckINFfiU" />
 </picture>
</a>

<a id="license"></a>
## 📜 License

This project is released under the Apache 2.0 license; see [LICENSE](LICENSE) for
details. The Qwen3-VL base model weights are governed by their own license, and
the ShareGPT4V images are restricted to academic use.

---

<p align="center">
  Built with ❤️ for efficient AI community
</p>

<p align="center">
  <a href="https://github.com/ForeverBlue816/GRACE/stargazers"><img src="https://img.shields.io/github/stars/ForeverBlue816/GRACE?style=social" alt="GitHub stars"/></a>
  <a href="https://github.com/ForeverBlue816/GRACE/network/members"><img src="https://img.shields.io/github/forks/ForeverBlue816/GRACE?style=social" alt="GitHub forks"/></a>
</p>
