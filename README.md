<p align="center">
  <img src="assets/swan.png" alt="GRACE" width="140"/>
</p>

<h1 align="center">GRACE</h1>

<p align="center"><b>Gated Relational Alignment via Confidence-based Distillation<br/>for Quantization-Aware Training of Vision–Language Models</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.22709"><img src="https://img.shields.io/badge/ICML%202026-Accepted-1f6feb?style=flat-square" alt="ICML 2026"/></a>
  <a href="https://arxiv.org/abs/2601.22709"><img src="https://img.shields.io/badge/arXiv-2601.22709-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"/></a>
  <a href="https://huggingface.co/FoeverBLUE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffce1c?style=flat-square" alt="Hugging Face Models"/></a>
  <img src="https://img.shields.io/badge/python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.5+"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-4caf50?style=flat-square" alt="License: Apache 2.0"/></a>
</p>

<p align="center"><b>Official PyTorch implementation of GRACE, accepted at ICML 2026.</b></p>

<p align="center">
  📄 <a href="https://arxiv.org/abs/2601.22709"><b>Paper</b></a>
  &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/FoeverBLUE"><b>Models</b></a>
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
- [📦 Deployment (LLaVA-1.5 INT4 / AWQ)](#deployment)
- [📝 Citation](#citation)
- [🙏 Acknowledgements](#acknowledgements)
- [📜 License](#license)

</details>

GRACE is a quantization-aware training (QAT) framework for vision–language
models that recovers most of the accuracy lost to low-bit weight quantization
by combining:

- **GDKD** — confidence-gated *Decoupled Knowledge Distillation* (TCKD + NCKD),
  with the trade-off `β` adapted online by an Information-Bottleneck controller.
- **RCKA** — *Relational Centered Kernel Alignment* on penultimate-layer
  visual tokens, aligning the student's relational geometry to the teacher's.
- **Group-wise LSQ QAT** — learned per-group weight scales (W4 / W8,
  group size 128) on the LLM and MLP projector, frozen ViT.

Training optimizes

```
L_total = L_CE  +  β · L_GDKD  +  ω · L_RCKA
```

with `β` driven by an IB controller (`τ`, `η`) and `ω` warmed up linearly.
Defaults follow the values in the paper (Table 6); see
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
(reference upper bound); all GRACE-Qwen3 variants are **2B** students. Best result
among the 2B Qwen3-VL models is in **bold**.

We release GRACE on Qwen3-VL here because it is the most current backbone and
gives a fairer, up-to-date point of comparison, with the vanilla
Qwen3-VL-2B-Instruct as the baseline. The paper itself reports GRACE on
LLaVA-1.5 and Qwen2-VL; we additionally release the LLaVA-1.5 W4G128 (INT4)
checkpoint from the paper in the model zoo below.

| Model | Params | Precision | HallB | MMBench | ScienceQA | AI2D | MMMU | SEED | MMStar | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-8B *(teacher, ref.)* | 8B | BF16 | 61.1 | 84.5 | 85.0 | 85.7 | 69.6 | 77.5 | 70.9 | 76.3 |
| Qwen3-VL-2B *(baseline)*      | 2B | BF16 | 51.4 | 78.4 | 81.4 | 76.9 | 53.4 | 71.2 | 58.3 | 67.3 |
| **Qwen3-VL-2B-GRACE**         | 2B | BF16 | **66.9** | **86.4** | **86.2** | **81.3** | **72.1** | **76.7** | **67.3** | **76.7** |
| Qwen3-VL-2B-GRACE (W8G128)    | 2B | INT8 | 66.1 | 85.5 | 85.3 | 80.4 | 71.3 | 75.9 | 66.5 | 75.9 |
| Qwen3-VL-2B-GRACE (W4G128)    | 2B | INT4 | 65.4 | 84.6 | 84.3 | 79.5 | 70.5 | 75.1 | 65.8 | 75.0 |

> GRACE lifts the Qwen3-VL-2B baseline by **+9.4 avg** and matches or slightly
> exceeds the 8B teacher on average (76.7 vs. 76.3) at roughly 1/4 the
> parameters. The W4G128 (INT4) model retains **98%** of the BF16 average.

---

<a id="model-zoo"></a>
## 🤗 Model Zoo

| Model | Backbone | Bits | Group | Checkpoint description | HF Hub |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-GRACE-BF16 | Qwen3-VL-2B | bf16 | — | Full-precision GRACE checkpoint; used as the student initialization for the W8/W4 Qwen3-VL runs. | [FoeverBLUE/Qwen3-VL-2B-GRACE-BF16](https://huggingface.co/FoeverBLUE/Qwen3-VL-2B-GRACE-BF16) |
| Qwen3-VL-2B-GRACE-W8G128 | Qwen3-VL-2B | int8 | 128 | INT8 QAT checkpoint with group size 128; high-retention quantized Qwen3-VL student. | [FoeverBLUE/Qwen3-VL-2B-GRACE-W8G128](https://huggingface.co/FoeverBLUE/Qwen3-VL-2B-GRACE-W8G128) |
| Qwen3-VL-2B-GRACE-W4G128 | Qwen3-VL-2B | int4 | 128 | INT4 QAT checkpoint with group size 128; compact Qwen3-VL release retaining about 98% of the BF16 average. | [FoeverBLUE/Qwen3-VL-2B-GRACE-W4G128](https://huggingface.co/FoeverBLUE/Qwen3-VL-2B-GRACE-W4G128) |
| LLaVA-1.5-7B-GRACE-W4G128 | LLaVA-1.5-7B | int4 | 128 | INT4 QAT checkpoint from the GRACE paper with learned scales; released for reproducing the LLaVA-1.5 experiments. | [FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128](https://huggingface.co/FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128) |

The BF16 Qwen3-VL checkpoint is the full-precision GRACE student used as the
initial student weights for the W8 and W4 Qwen3-VL runs. The LLaVA-1.5 W4G128
checkpoint corresponds to the paper setting and includes GRACE-specific QAT
quantized weights for reproducing the INT4 LLaVA experiments.

**Quick load:**

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ckpt = "FoeverBLUE/Qwen3-VL-2B-GRACE-W4G128"
model = Qwen3VLForConditionalGeneration.from_pretrained(ckpt, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(ckpt)
```

---

<a id="repository-layout"></a>
## 📁 Repository Layout

```
.
├── qwen-vl-finetune/        # Training entry points (SFT, QAT, GRACE)
│   ├── qwenvl/
│   │   ├── data/            # Dataset registry + LLaVA-style loader
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
│       └── finetune_qwen3vl_2b_grace.slurm  # GRACE
├── evaluation/              # lmms-eval driver + per-benchmark configs
├── deployment/              # LLaVA-1.5 tree: QAT training + AWQ INT4 packing/inference
│   ├── llava/quantize/      # QAT→AWQ conversion + WQLinear_GEMM kernels
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

GRACE was trained on 16*A100-64GB GPUs. The
reference SLURM scripts pin the host toolchain in their `module load` block:

| Component | Version |
| --- | --- |
| CUDA driver / runtime (host) | **12.3** |
| GCC                          | **12.2.0** |
| Python                       | **3.11** |
| PyTorch                      | **2.5.1** (cu121 wheels — forward-compatible with CUDA 12.3 driver) |
| flash-attn                   | **2.7.2.post1** |
| DeepSpeed                    | **0.15.4** (ZeRO-2) |
| transformers                 | **5.9.0** |
| accelerate                   | **1.13.0** |

A frozen export of the full virtual environment is in
[requirements.txt](requirements.txt).

### Build the venv from scratch

```bash
# 1) System modules (Leonardo example — adapt to your cluster)
module purge
module load profile/deeplrn
module load cuda/12.3
module load gcc/12.2.0

# 2) Create venv
python3.11 -m venv ${HOME}/qwen3vl
source ${HOME}/qwen3vl/bin/activate
pip install -U pip wheel setuptools

# 3) PyTorch + CUDA runtime (cu121 wheels)
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 4) Everything else (pinned to the released training env)
pip install -r requirements.txt

# 5) flash-attn — must build AFTER torch is installed
pip install flash-attn==2.7.2.post1 --no-build-isolation

# 6) Local utility package (image / video preprocessing for Qwen3-VL)
pip install -e qwen-vl-utils/
```

### Compute-node environment variables

Pre-stage models on the login node (or any internet-reachable host):

```bash
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
    --local-dir ${SCRATCH_ROOT}/Qwen3-VL-2B-Instruct
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir ${SCRATCH_ROOT}/Qwen3-VL-8B-Instruct
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

`sharegpt4v_mix665k_*` is the main SFT mix used in the paper;
`sharegpt4v_instruct_gpt4-vision_cap100k.json` is the original GPT-4V
caption set.

### 1. Download annotation JSONs

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
Download and unpack the following sources (only the ones the JSONs actually
reference are required):

| Source | URL |
| --- | --- |
| LAION-CC-SBU-558K | [images.zip](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip) |
| COCO              | [train2017.zip](http://images.cocodataset.org/zips/train2017.zip) |
| SAM (subset)      | [segment-anything-downloads](https://ai.meta.com/datasets/segment-anything-downloads/) — `000000~000050.tar`. For SFT-only you can take the 9k subset [here](https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link). |
| GQA               | [images.zip](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) |
| OCR-VQA           | [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) — save all as `.jpg` |
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
[qwen-vl-finetune/qwenvl/data/__init__.py](qwen-vl-finetune/qwenvl/data/__init__.py)
— it reads `SHAREGPT4V_ROOT` from the environment.

---

<a id="training"></a>
## 🚀 Training

### 1. BF16 SFT baseline (optional — also our released `*-BF16` checkpoint)

```bash
sbatch qwen-vl-finetune/scripts/finetune_qwen3vl_2b_bf16.slurm
```

### 2. QAT-only baseline (ablation of GRACE without distillation)

```bash
# W4 G128
sbatch qwen-vl-finetune/scripts/finetune_qwen3vl_2b_qat.slurm
# W8 G128
sbatch --export=ALL,QAT_BITS=8 qwen-vl-finetune/scripts/finetune_qwen3vl_2b_qat.slurm
```

### 3. GRACE (full method)

```bash
# W4 G128 — produces FoeverBLUE/Qwen3-VL-2B-GRACE-W4G128
sbatch qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace.slurm

# W8 G128 — produces FoeverBLUE/Qwen3-VL-2B-GRACE-W8G128
sbatch --export=ALL,QAT_BITS=8 qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace.slurm
```

Common env-var overrides (all scripts):

| Variable | Default | Description |
| --- | --- | --- |
| `SHAREGPT4V_ROOT` | `PATH_TO_SHAREGPT4V_ROOT` | Root of ShareGPT4V tree. |
| `DATASETS` | `sharegpt4v_mix665k` | Comma-separated, `%NN` suffix downsamples. |
| `MODEL_NAME_OR_PATH` | `${SCRATCH_ROOT}/Qwen3-VL-2B-Instruct` | Student init. |
| `TEACHER_MODEL_PATH` | `${SCRATCH_ROOT}/Qwen3-VL-8B-Instruct` | GRACE only. |
| `QAT_BITS` / `QAT_GROUP_SIZE` | `4` / `128` | LSQ fake-quant config. |
| `OUTPUT_DIR` | `${CKPT_ROOT}/${RUN_NAME}` | Auto-resumes from latest `checkpoint-*`. |

GRACE-specific knobs (defaults follow the paper):

| Variable | Default | Meaning |
| --- | --- | --- |
| `DKD_TEMPERATURE` | `2.0` | KD temperature `T`. |
| `DKD_ALPHA` / `DKD_BETA` | `1.0` / `4.0` | TCKD / NCKD weights. |
| `RCKA_WEIGHT` | `3.0` | `ω` for L_RCKA. |
| `RCKA_LAYER` | `-2` | Hidden-state index for RCKA. |
| `IB_TAU` / `IB_ETA` | `3.0` / `0.003` | IB controller target / step size. |
| `IB_BETA_INIT/MIN/MAX` | `0.5` / `0.1` / `1.0` | `β` schedule bounds. |
| `RCKA_WARMUP_STEPS` | `400` | Linear warmup for RCKA. |

The full reference run uses 4 × 4 × A100-80GB, DeepSpeed ZeRO-2, BF16,
effective batch 512. Adjust `--nodes`, `PER_DEVICE_BATCH`, and `GRAD_ACCUM`
to fit your cluster.

---

<a id="evaluation"></a>
## 📈 Evaluation

We score every checkpoint with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
Per-benchmark configs live under
[evaluation/](https://github.com/ForeverBlue816/GRACE/tree/main/evaluation).

```bash
# Example: evaluate the released W4 checkpoint on ScienceQA
sbatch --export=ALL,MODEL=FoeverBLUE/Qwen3-VL-2B-GRACE-W4G128 \
       evaluation/ScienceQA/eval_scienceqa.slurm
```

---

<a id="deployment"></a>
## 📦 Deployment — LLaVA-1.5 INT4 (AWQ)

The paper's LLaVA-1.5-7B GRACE results ship with a deployable **real INT4** build.
A GRACE QAT checkpoint stores BF16 weights snapped onto the INT4 grid plus a
`qat_quantized_weights.bin` sidecar (the learned per-group scales). For deployment
we **pack** those layers into genuine 4-bit AutoAWQ tensors
(`qweight` / `qzeros` / `scales`) that run through AWQ's INT4 GEMM kernels. The
packing is **bit-exact** (the integer codes are unchanged; only the per-group
scales are stored in FP16) and shrinks the language-model weights from
**≈14.2 GB (BF16) → ≈4.6 GB (~3.1× smaller)**.

Two LLaVA-1.5 checkpoints are released:

| Repo | What it stores | Use it for |
| --- | --- | --- |
| [LLaVA-1.5-7B-GRACE-W4G128](https://huggingface.co/FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128) | BF16 weights on the INT4 grid **+ `qat_quantized_weights.bin`** sidecar | re-packing / research; the source for the conversion below |
| [LLaVA-1.5-7B-GRACE-W4G128-AWQ](https://huggingface.co/FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128-AWQ) | real packed `qweight` / `qzeros` / `scales` | drop-in INT4 inference |

The LLaVA-1.5 deployment code lives under [deployment/](deployment) (a vendored
LLaVA-1.5 tree with the GRACE QAT + AWQ additions). It needs its **own**
`transformers==4.37.2` environment, separate from the `qwen3vl` training venv.
**Run every command below from the `deployment/` directory.**

### 1. Environment

LLaVA-1.5 needs its **own** environment pinned to `transformers==4.37.2` (keep it
separate from the `qwen3vl` training venv). The exact tested versions are frozen in
[deployment/requirements.txt](deployment/requirements.txt) for a one-shot install:

```bash
cd deployment

# fresh venv for the LLaVA-1.5 stack (do NOT reuse the qwen3vl venv)
python3 -m venv ~/llava && source ~/llava/bin/activate
pip install -U pip

pip install -r requirements.txt   # exact tested pins (torch cu121, transformers 4.37.2, …)
pip install -e . --no-deps        # register the local `llava` package

# OPTIONAL speedups (build AFTER torch is installed):
pip install flash-attn==2.5.8 --no-build-isolation
pip install autoawq-kernels       # fused INT4 GEMM kernels — large speedup; without
                                  # them the model still runs via a PyTorch dequant path
```

> Tested on an A100 (CUDA 12.x driver) with the versions in `requirements.txt`.
> `torch==2.1.2+cu121` bundles its own CUDA runtime, so a system CUDA toolkit + GCC
> are only needed to build `flash-attn` / `autoawq-kernels`.

### 2. Run the released AWQ model (quickest path)

Download the packed checkpoint and run one-shot inference on the bundled
[chinaairlines.jpg](deployment/images/chinaairlines.jpg):

```bash
# download the packed INT4 model
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128-AWQ"))
PY

python scripts/deploy_awq_llava.py \
    --load-packed /path/to/LLaVA-1.5-7B-GRACE-W4G128-AWQ \
    --image-file images/chinaairlines.jpg \
    --query "Please describe the scene in the picture in detail." \
    --conv-mode vicuna_v1 \
    --max-new-tokens 256
```

On the bundled [chinaairlines.jpg](deployment/images/chinaairlines.jpg) this prints:

<p align="center">
  <img src="deployment/images/chinaairlines.jpg" alt="chinaairlines.jpg example" width="55%"/>
</p>

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

The packed INT4 model runs the whole thing in **≈5.9 GB** of GPU memory. The
≈4.3 tok/s above is the pure-PyTorch dequant fallback — install `autoawq-kernels`
(see Environment) for the fused INT4 kernels and a large speedup.

`--load-packed` reuses LLaVA's normal loader for the architecture / CLIP vision
tower / tokenizer, rebuilds the AWQ modules listed in `awq_quantized_modules.json`,
and loads the packed tensors — no re-packing. The CLIP vision tower is pulled from
`openai/clip-vit-large-patch14-336` (via the `mm_vision_tower` field), so the host
needs either internet on first run or a local CLIP path.

> **Note:** this is the original LLaVA-1.5 architecture with AWQ-packed weights, so
> it loads through this codebase — a plain `from_pretrained` will **not**
> reconstruct the INT4 layers.

### 3. Convert a BF16 QAT checkpoint → AWQ yourself

To reproduce the AWQ build from the released QAT checkpoint (or to pack one you
trained), load the BF16 fake-quant checkpoint, pack it into real 4-bit, and persist
it with `--save-dir`:

```bash
# download the QAT (fake-quant BF16 + sidecar) checkpoint
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("FoeverBLUE/LLaVA-1.5-7B-GRACE-W4G128"))
PY

# load fp16 → pack to real INT4 → run → persist the packed model
python scripts/deploy_awq_llava.py \
    --model-path /path/to/LLaVA-1.5-7B-GRACE-W4G128 \
    --image-file images/chinaairlines.jpg \
    --query "Please describe the scene in the picture in detail." \
    --conv-mode vicuna_v1 \
    --max-new-tokens 256 \
    --save-dir ./checkpoints/llava-w4-awq-packed
```

This loads the BF16 checkpoint, reads its `qat_quantized_weights.bin` sidecar, swaps
every quantized LLM linear for an AWQ `WQLinear_GEMM`, verifies the packing is
bit-exact, runs inference, and writes the packed model (plus
`awq_quantized_modules.json`) to `--save-dir`. From then on reload it instantly with
`--load-packed ./checkpoints/llava-w4-awq-packed` — the `--save-dir` of one run is
exactly the `--load-packed` of the next. Drop `--image-file` and add `--text-only`
for a fast LLM-only smoke test.

<details>
<summary><b>How the conversion works (symmetric LSQ-QAT → asymmetric AWQ)</b></summary>

GRACE QAT is **symmetric signed** per group: code `q ∈ [-8, 7]`, a per-group scale
`s`, **no zero point**, so the dequantized weight is `W = q · s` (groups of
`group_size = 128` along the input dim). AWQ's GEMM kernel is **asymmetric
unsigned**: `W = scales · (q_awq − zeros)` with `q_awq ∈ [0, 15]`. The two line up
*exactly* with a constant zero-point:

```
zeros  = 8  (= 2^(bits-1))
scales = s  (= exp(log_scale), stored FP16)
q_awq  = q + 8 ∈ [0, 15]
⇒  scales · (q_awq − 8) = s · q = W      (no error beyond FP16 scale rounding)
```

Only the per-group scales change dtype; the integer codes are identical. The
converter ([deployment/llava/quantize/qat_to_awq.py](deployment/llava/quantize/qat_to_awq.py))
reads the sidecar as the source of truth for which layers were quantized, packs
each one, and (unless `--no-verify`) asserts the max int-code mismatch is `0`. Only
the LLM linears (`self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj` across all
decoder layers) are quantized; the CLIP vision tower, `mm_projector`, embeddings,
`lm_head`, and norms stay FP16.

</details>

### Load it programmatically

```python
import os, glob, json
from safetensors.torch import load_file
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.quantize import build_awq_skeleton

d = "/path/to/LLaVA-1.5-7B-GRACE-W4G128-AWQ"
meta = json.load(open(os.path.join(d, "awq_quantized_modules.json")))

tokenizer, model, image_processor, _ = load_pretrained_model(
    d, None, get_model_name_from_path(d), device_map="cuda", device="cuda")

# replace the LLM linears with AWQ modules, then load the packed weights
build_awq_skeleton(model, meta["modules"], bits=meta["bits"],
                   group_size=meta["group_size"], device="cuda")
sd = {}
for f in glob.glob(os.path.join(d, "*.safetensors")):
    sd.update(load_file(f))
prefixes = tuple(n + "." for n in meta["modules"])
model.load_state_dict({k: v for k, v in sd.items() if k.startswith(prefixes)}, strict=False)
model.eval()
```

---

<a id="citation"></a>
## 📝 Citation

If you use GRACE or the released checkpoints in your research, please cite:

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
The ShareGPT4V training data is from
[Lin-Chen/ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V).
Evaluation is powered by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

<a id="license"></a>
## 📜 License

This project is released under the Apache 2.0 license — see [LICENSE](LICENSE).
The Qwen3-VL base model weights are governed by their own license; the
ShareGPT4V images are restricted to academic use.
