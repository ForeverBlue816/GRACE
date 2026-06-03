"""Deploy a group-wise INT4 QAT LLaVA checkpoint as a REAL AutoAWQ-packed model.

It loads the (fake-quant / baked) checkpoint with LLaVA's normal loader, then
swaps every QAT-quantized nn.Linear in the language model for an AutoAWQ
`WQLinear_GEMM` whose weights are genuinely stored as packed 4-bit integers
(scales + zero-point). Inference then runs through AWQ's INT4 GEMM kernels
(`awq_ext` from `autoawq-kernels`) when available, otherwise a correct but
slower pure-PyTorch dequant path.

Examples
--------
# 1) text-only smoke test (fast, no image / vision tower needed for the LLM):
python scripts/deploy_awq_llava.py \
    --model-path /path/to/LLaVA-1.5-7B-GRACE-W4G128 \
    --text-only --query "Explain what 4-bit weight quantization is."

# 2) real multimodal inference:
python scripts/deploy_awq_llava.py \
    --model-path /path/to/LLaVA-1.5-7B-GRACE-W4G128 \
    --image-file images/chinaairlines.jpg --query "Describe this image in detail."

# 3) also persist the packed checkpoint for later (reload with --load-packed):
python scripts/deploy_awq_llava.py --model-path /path/to/LLaVA-1.5-7B-GRACE-W4G128 \
    --text-only --query "hi" --save-dir ./checkpoints/llava-w4-awq-packed
"""

import argparse
import os
import sys
import time

import torch

# make `llava` importable when run as a plain script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.quantize import convert_llava_to_awq, build_awq_skeleton


def _fmt_gb(n_bytes):
    return f"{n_bytes / 1e9:.2f} GB"


def _param_buffer_bytes(model):
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        total += t.numel() * t.element_size()
    return total


def load_packed_awq_llava(packed_dir, device):
    """Load a previously --save-dir'd AWQ-packed LLaVA WITHOUT re-packing.

    Reuses LLaVA's own loader for the architecture / vision tower / tokenizer /
    image processor (the 4-bit LLM linears show up as 'missing weight' there,
    since the checkpoint stores qweight/qzeros/scales), then replaces those
    modules with WQLinear_GEMM and copies in the packed tensors.
    """
    import glob
    import json
    import gc
    from safetensors.torch import load_file

    meta_path = os.path.join(packed_dir, "awq_quantized_modules.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found — this dir was not produced by --save-dir.\n"
            f"Use the convert path instead: --model-path <ckpt> [--qat-bin ...].")
    meta = json.load(open(meta_path))
    names, bits, gs = meta["modules"], meta["bits"], meta["group_size"]

    # Strip a HF `quantization_config` if present: it would make from_pretrained
    # try to import the (broken) autoawq package. We apply AWQ manually below.
    cfg_path = os.path.join(packed_dir, "config.json")
    cfg = json.load(open(cfg_path))
    if cfg.pop("quantization_config", None) is not None:
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        print("[load-packed] removed HF quantization_config from config.json (loading AWQ manually)")

    name = get_model_name_from_path(packed_dir)
    print(f"[load-packed] loading skeleton + vision tower from {packed_dir} ...")
    # The packed checkpoint stores qweight/qzeros/scales instead of `weight`, so
    # from_pretrained treats the ~6.5B LLM weight matrices as "missing keys" and
    # randomly initializes them — pure waste, since we overwrite those modules
    # with WQLinear_GEMM immediately after. Disable that init to avoid a slow,
    # throwaway random initialization of the whole language model.
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    _orig_init = LlavaLlamaForCausalLM._init_weights
    LlavaLlamaForCausalLM._init_weights = lambda self, module: None
    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            packed_dir, None, name, device_map=device, device=device)
    finally:
        LlavaLlamaForCausalLM._init_weights = _orig_init

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
    return tokenizer, model, image_processor


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
    ap.add_argument("--low-vram", action="store_true",
                    help="load + pack on CPU first, then move to --device (use if fp16 model does not fit)")
    ap.add_argument("--no-verify", action="store_true", help="skip the bit-exactness check (faster)")

    ap.add_argument("--text-only", action="store_true", help="run a text-only prompt (no image)")
    ap.add_argument("--image-file", default=None)
    ap.add_argument("--query", required=True)
    ap.add_argument("--conv-mode", default="llava_v1")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--save-dir", default=None, help="optionally save the packed model here")
    args = ap.parse_args()

    if not (args.model_path or args.load_packed):
        ap.error("pass either --model-path (convert & run) or --load-packed (reuse a saved pack)")

    disable_torch_init()

    if args.load_packed:
        # -------- fast path: load an already-packed model, no re-packing -------
        tokenizer, model, image_processor = load_packed_awq_llava(args.load_packed, args.device)
        print(f"[deploy] AWQ-packed footprint: {_fmt_gb(_param_buffer_bytes(model))}")
    else:
        # -------- convert path: load fp16 then pack in memory ------------------
        qat_bin = args.qat_bin or os.path.join(args.model_path, "qat_quantized_weights.bin")
        if not os.path.isfile(qat_bin):
            raise FileNotFoundError(f"QAT sidecar not found: {qat_bin}")
        model_name = get_model_name_from_path(args.model_path)

        load_device = "cpu" if args.low_vram else args.device
        print(f"[deploy] loading fp16 checkpoint on {load_device} ...")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, None, model_name, device_map=load_device, device=load_device,
        )
        fp16_bytes = _param_buffer_bytes(model)
        print(f"[deploy] fp16 footprint (params+buffers): {_fmt_gb(fp16_bytes)}")

        # ---- the actual quantization: swap to real packed 4-bit kernels ------
        quant_names = convert_llava_to_awq(
            model, qat_bin, bits=args.bits, group_size=args.group_size, verify=not args.no_verify,
        )
        awq_bytes = _param_buffer_bytes(model)
        print(f"[deploy] AWQ-packed footprint: {_fmt_gb(awq_bytes)}  "
              f"({fp16_bytes / max(awq_bytes, 1):.2f}x smaller)")

        if args.low_vram and args.device != "cpu":
            print(f"[deploy] moving packed model to {args.device} ...")
            model = model.to(args.device)
            if hasattr(model.get_model(), "vision_tower"):
                model.get_vision_tower().to(args.device, dtype=torch.float16)
        model.eval()

        if args.save_dir:
            import json
            print(f"[deploy] saving packed model to {args.save_dir}")
            os.makedirs(args.save_dir, exist_ok=True)
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            # list of quantized modules -> needed to rebuild the skeleton on reload
            with open(os.path.join(args.save_dir, "awq_quantized_modules.json"), "w") as f:
                json.dump({"bits": args.bits, "group_size": args.group_size,
                           "modules": quant_names}, f, indent=2)
            print(f"[deploy] saved. Reload it directly with: --load-packed {args.save_dir}")

    # ---------------------------- inference -----------------------------------
    qs = args.query
    if not args.text_only:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor, image_sizes = None, None
    if not args.text_only:
        from PIL import Image
        image = Image.open(args.image_file).convert("RGB")
        image_sizes = [image.size]
        images_tensor = process_images([image], image_processor, model.config).to(
            model.device, dtype=torch.float16
        )

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    dt = time.time() - t0
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("\n=================== OUTPUT ===================")
    print(text)
    print("==============================================")
    print(f"[deploy] generated in {dt:.2f}s ({args.max_new_tokens / max(dt, 1e-6):.1f} tok/s nominal)")
    if args.device.startswith("cuda"):
        print(f"[deploy] peak GPU mem during generate: {_fmt_gb(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    main()
