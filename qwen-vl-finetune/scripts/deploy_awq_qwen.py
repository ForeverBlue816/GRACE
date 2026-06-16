"""Deploy a group-wise INT4 QAT Qwen3-VL checkpoint as a REAL AutoAWQ-packed model.

It loads the (fake-quant / baked) checkpoint with the normal HF loader
(`Qwen3VLForConditionalGeneration`), then swaps every QAT-quantized nn.Linear in
the language model for an AutoAWQ `WQLinear_GEMM` whose weights are genuinely
stored as packed 4-bit integers (scales + zero-point). Inference then runs
through AWQ's INT4 GEMM kernels (`awq_ext` from `autoawq-kernels`) when available,
otherwise a correct but slower pure-PyTorch dequant path.

The conversion reads `qat_quantized_weights.bin` (the training sidecar) as the
single source of truth for which layers were quantized and their per-group
log-scales, so it is bit-exact w.r.t. the trained INT4 grid.

Examples
--------
# 1) text-only smoke test (fast, no image / vision tower needed for the LLM):
python scripts/deploy_awq_qwen.py \
    --model-path /path/to/qwen3vl-2b-qat-w4g128 \
    --text-only --query "Explain what 4-bit weight quantization is."

# 2) real multimodal inference:
python scripts/deploy_awq_qwen.py \
    --model-path /path/to/qwen3vl-2b-qat-w4g128 \
    --image /path/to/img.jpg --query "Describe this image in detail."

# 3) convert once, persist the packed checkpoint, then reload it instantly later:
python scripts/deploy_awq_qwen.py --model-path /path/to/qwen3vl-2b-qat-w4g128 \
    --text-only --query "hi" --save-dir ./checkpoints/qwen3vl-2b-w4-awq-packed
python scripts/deploy_awq_qwen.py --load-packed ./checkpoints/qwen3vl-2b-w4-awq-packed \
    --image /path/to/img.jpg --query "Describe this image in detail."
"""

import argparse
import gc
import glob
import json
import os
import sys
import time

import torch

# make `qwenvl` importable when run as a plain script (scripts/ -> qwen-vl-finetune/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwenvl.quantize import convert_to_awq, build_awq_skeleton


_DTYPES = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16}


def _fmt_gb(n_bytes):
    return f"{n_bytes / 1e9:.2f} GB"


def _param_buffer_bytes(model):
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        total += t.numel() * t.element_size()
    return total


def load_packed_awq_qwen(packed_dir, device, dtype):
    """Load a previously --save-dir'd AWQ-packed Qwen3-VL WITHOUT re-packing.

    Reuses the HF loader for the architecture / vision tower / processor (the
    4-bit LLM linears show up as 'missing weight' there, since the checkpoint
    stores qweight/qzeros/scales), then replaces those modules with
    WQLinear_GEMM and copies in the packed tensors.
    """
    from safetensors.torch import load_file

    meta_path = os.path.join(packed_dir, "awq_quantized_modules.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found — this dir was not produced by --save-dir.\n"
            f"Use the convert path instead: --model-path <ckpt> [--qat-bin ...].")
    meta = json.load(open(meta_path))
    names, bits, gs = meta["modules"], meta["bits"], meta["group_size"]

    # A stray HF `quantization_config` would make from_pretrained try to import
    # the autoawq package. We apply AWQ manually, so strip it if present.
    cfg_path = os.path.join(packed_dir, "config.json")
    cfg = json.load(open(cfg_path))
    if cfg.pop("quantization_config", None) is not None:
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        print("[load-packed] removed HF quantization_config from config.json (loading AWQ manually)")

    print(f"[load-packed] loading skeleton + vision tower from {packed_dir} ...")
    # The packed checkpoint stores qweight/qzeros/scales instead of `weight`, so
    # from_pretrained treats the quantized LLM weight matrices as "missing keys"
    # and would randomly initialize them — pure waste, since we overwrite those
    # modules with WQLinear_GEMM immediately after. Disable that init.
    _orig_init = Qwen3VLForConditionalGeneration._init_weights
    Qwen3VLForConditionalGeneration._init_weights = lambda self, module: None
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            packed_dir, torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True)
    finally:
        Qwen3VLForConditionalGeneration._init_weights = _orig_init
    processor = AutoProcessor.from_pretrained(packed_dir)

    print(f"[load-packed] rebuilding {len(names)} AWQ modules and loading packed weights ...")
    build_awq_skeleton(model, names, bits=bits, group_size=gs, device=device)
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    sd = {}
    for shard in sorted(glob.glob(os.path.join(packed_dir, "*.safetensors"))):
        sd.update(load_file(shard))
    prefixes = tuple(n + "." for n in names)
    quant_sd = {k: v for k, v in sd.items() if k.startswith(prefixes)}
    _, unexpected = model.load_state_dict(quant_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} packed tensors did not match the "
                           f"skeleton, e.g. {list(unexpected)[:3]}")
    sample = model.get_submodule(names[0])
    if int(sample.qweight.abs().sum().item()) == 0:
        raise RuntimeError("AWQ buffers are still empty after load — weights not applied.")
    print(f"[load-packed] applied {len(quant_sd)} packed tensors into {len(names)} modules.")
    model.eval()
    return model, processor


def _save_packed(model, processor, src_dir, save_dir, bits, group_size, quant_names):
    print(f"[deploy] saving packed model to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    # list of quantized modules -> needed to rebuild the skeleton on reload
    with open(os.path.join(save_dir, "awq_quantized_modules.json"), "w") as f:
        json.dump({"bits": bits, "group_size": group_size, "modules": quant_names}, f, indent=2)
    # carry over any aux files the processor/model save may not emit
    import shutil
    for aux in ("chat_template.jinja", "preprocessor_config.json",
                "video_preprocessor_config.json", "generation_config.json"):
        s, d = os.path.join(src_dir, aux), os.path.join(save_dir, aux)
        if os.path.isfile(s) and not os.path.isfile(d):
            shutil.copy2(s, d)
    print(f"[deploy] saved. Reload it directly with: --load-packed {save_dir}")


def run_inference(model, processor, args):
    content = []
    if not args.text_only:
        content.append({"type": "image", "image": args.image})
    content.append({"type": "text", "text": args.query})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.text_only:
        inputs = processor(text=[text], return_tensors="pt")
    else:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    if str(model.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    dt = time.time() - t0
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    out_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    n_new = sum(t.numel() for t in trimmed)
    print("\n=================== OUTPUT ===================")
    print(out_text)
    print("==============================================")
    print(f"[deploy] generated {n_new} tokens in {dt:.2f}s ({n_new / max(dt, 1e-6):.1f} tok/s)")
    if str(model.device).startswith("cuda"):
        print(f"[deploy] peak GPU mem during generate: {_fmt_gb(torch.cuda.max_memory_allocated())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None, help="QAT checkpoint dir (convert-and-run path)")
    ap.add_argument("--load-packed", default=None,
                    help="load an already --save-dir'd AWQ-packed model directly (skips re-packing)")
    ap.add_argument("--qat-bin", default=None,
                    help="path to qat_quantized_weights.bin (default: <model-path>/qat_quantized_weights.bin)")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=list(_DTYPES),
                    help="compute dtype for the non-quantized parts (vision tower, embeddings, ...)")
    ap.add_argument("--no-verify", action="store_true", help="skip the bit-exactness check (faster)")

    ap.add_argument("--text-only", action="store_true", help="run a text-only prompt (no image)")
    ap.add_argument("--image", default=None, help="image path or URL")
    ap.add_argument("--query", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--save-dir", default=None, help="optionally save the packed model here")
    args = ap.parse_args()

    if not (args.model_path or args.load_packed):
        ap.error("pass either --model-path (convert & run) or --load-packed (reuse a saved pack)")
    if not args.text_only and not args.image:
        ap.error("pass --image <path|url> for multimodal inference, or --text-only")

    dtype = _DTYPES[args.dtype]

    if args.load_packed:
        # -------- fast path: load an already-packed model, no re-packing -------
        model, processor = load_packed_awq_qwen(args.load_packed, args.device, dtype)
        print(f"[deploy] AWQ-packed footprint: {_fmt_gb(_param_buffer_bytes(model))}")
    else:
        # -------- convert path: load bf16 then pack in memory ------------------
        qat_bin = args.qat_bin or os.path.join(args.model_path, "qat_quantized_weights.bin")
        if not os.path.isfile(qat_bin):
            raise FileNotFoundError(f"QAT sidecar not found: {qat_bin}")

        print(f"[deploy] loading checkpoint on {args.device} ({args.dtype}) ...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=dtype, device_map=args.device, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(args.model_path)
        bf16_bytes = _param_buffer_bytes(model)
        print(f"[deploy] dense footprint (params+buffers): {_fmt_gb(bf16_bytes)}")

        # ---- the actual quantization: swap to real packed 4-bit kernels ------
        quant_names = convert_to_awq(
            model, qat_bin, bits=args.bits, group_size=args.group_size, verify=not args.no_verify)
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        awq_bytes = _param_buffer_bytes(model)
        print(f"[deploy] AWQ-packed footprint: {_fmt_gb(awq_bytes)}  "
              f"({bf16_bytes / max(awq_bytes, 1):.2f}x smaller)")
        model.eval()

        if args.save_dir:
            _save_packed(model, processor, args.model_path, args.save_dir,
                         args.bits, args.group_size, quant_names)

    run_inference(model, processor, args)


if __name__ == "__main__":
    main()
