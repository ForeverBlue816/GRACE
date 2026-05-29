"""MMBench evaluation (dev split) for a fine-tuned Qwen3-VL checkpoint.

MMBench is multi-choice (2-4 options per question). The dev TSV ships gold
answers inline, with images base64-encoded into the `image` column — same
format VLMEvalKit/OpenCompass distribute.

Per-row accuracy is reported here. The full MMBench leaderboard metric is the
"circular evaluation" (CircularEval), which requires the model to answer the
same question correctly under every cyclic permutation of options — that's a
post-hoc reduction over per-row results and can be added on top of the JSONL
output if needed.

Single-process or sharded over GPUs via torchrun. Each rank writes a JSONL
shard; rank 0 merges and prints metrics at the end.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from collections import defaultdict

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

LETTERS = "ABCD"


def _decode_image_b64(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def load_data(tsv_path: str):
    df = pd.read_csv(tsv_path, sep="\t")
    return df


def collect_options(row):
    opts = []
    for letter in LETTERS:
        val = row.get(letter, None)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        opts.append((letter, str(val)))
    return opts


def format_prompt(question: str, hint, options) -> str:
    parts = []
    if hint is not None and not (isinstance(hint, float) and pd.isna(hint)):
        h = str(hint).strip()
        if h and h.lower() != "nan":
            parts.append(f"Context: {h}")
    parts.append(f"Question: {str(question).strip()}")
    opt_str = "\n".join(f"({l}) {t}" for l, t in options)
    parts.append(f"Options:\n{opt_str}")
    parts.append("Answer with only the option letter.")
    return "\n".join(parts)


def parse_choice(response: str, valid_letters) -> str:
    valid = set(valid_letters)
    for ch in response.strip().upper():
        if ch in valid:
            return ch
    return ""


def build_inputs(processor, image, prompt_text, device):
    content = [{"type": "image"}, {"type": "text", "text": prompt_text}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    return inputs.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tsv", required=True,
                    help="Path to MMBench dev TSV (e.g. mmbench_dev_en_20231003.tsv).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[init] world_size={world_size} model={args.model_path}", flush=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation=args.attn,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    df = load_data(args.tsv)
    if args.limit > 0:
        df = df.head(args.limit)
    rows = df.to_dict(orient="records")
    shard = rows[rank::world_size]
    if rank == 0:
        print(f"[data] total={len(rows)} shard={len(shard)}", flush=True)

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    shard_path = f"{args.output}.rank{rank}.jsonl"

    t0 = time.time()
    with open(shard_path, "w") as fout:
        for row in tqdm(shard, disable=(rank != 0), desc=f"rank{rank}"):
            try:
                options = collect_options(row)
                valid_letters = [l for l, _ in options]
                gold = str(row.get("answer", "")).strip().upper()
                image = _decode_image_b64(row["image"])

                prompt = format_prompt(row["question"], row.get("hint"), options)
                inputs = build_inputs(processor, image, prompt, device)
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                gen = out[0][inputs["input_ids"].shape[1]:]
                raw = processor.tokenizer.decode(gen, skip_special_tokens=True)
                pred = parse_choice(raw, valid_letters)
                err = None
            except Exception as e:
                raw, pred, gold, err = "", "", "", repr(e)

            fout.write(json.dumps({
                "index": row.get("index"),
                "pred": pred,
                "gold": gold,
                "correct": (pred != "" and pred == gold),
                "raw": raw,
                "category": row.get("category"),
                "l2_category": row.get("l2-category"),
                "n_options": len(valid_letters) if err is None else 0,
                "error": err,
            }, ensure_ascii=False) + "\n")
            fout.flush()

    if rank == 0:
        print(f"[rank0 done] elapsed={(time.time()-t0)/60:.1f} min", flush=True)

    if is_dist:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
    if rank != 0:
        return

    all_results = []
    for r in range(world_size):
        with open(f"{args.output}.rank{r}.jsonl") as f:
            for line in f:
                all_results.append(json.loads(line))

    n = len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    parse_fail = sum(1 for r in all_results if r["pred"] == "" and r["error"] is None)
    errors = sum(1 for r in all_results if r["error"] is not None)

    by_l2 = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    for r in all_results:
        k = r.get("l2_category") or "_unknown"
        by_l2[k][0] += int(r["correct"])
        by_l2[k][1] += 1
        c = r.get("category") or "_unknown"
        by_cat[c][0] += int(r["correct"])
        by_cat[c][1] += 1

    def fmt(d):
        return {k: {"acc": (c / n_), "correct": c, "total": n_}
                for k, (c, n_) in sorted(d.items())}

    summary = {
        "model_path": args.model_path,
        "tsv": args.tsv,
        "total": n,
        "correct": correct,
        "overall_acc": correct / n if n else 0.0,
        "parse_fail": parse_fail,
        "errors": errors,
        "by_l2_category": fmt(by_l2),
        "by_category": fmt(by_cat),
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)

    print("\n========== MMBench Eval ==========")
    print(f"Model:      {args.model_path}")
    print(f"TSV:        {args.tsv}")
    print(f"Total:      {n}")
    print(f"Correct:    {correct}")
    print(f"Accuracy:   {correct/n*100:.2f}%")
    print(f"Parse fail: {parse_fail}")
    print(f"Errors:     {errors}")
    print("\nBy l2-category:")
    for k, v in summary["by_l2_category"].items():
        print(f"  {k:24s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
