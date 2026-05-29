"""MMStar evaluation for a fine-tuned Qwen3-VL checkpoint.

MMStar is a 1500-sample 4-way multi-choice benchmark distributed by
VLMEvalKit/OpenCompass. The TSV ships gold answers inline with images
base64-encoded into the `image` column. Each sample carries a coarse
`category` (6 in total) and a finer `l2_category` (18 in total); the
official leaderboard reports per-category accuracy plus an overall average.

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

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

LETTERS = "ABCD"


def _decode_image_b64(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def load_data(tsv_path: str) -> pd.DataFrame:
    return pd.read_csv(tsv_path, sep="\t")


def collect_options(row):
    """MMStar's TSV may or may not have separate A/B/C/D columns. When it
    does (MMBench-style), build the option list. When it doesn't (the more
    common case — choices are already embedded in the `question` text),
    return an empty list and trust the question text."""
    opts = []
    for letter in LETTERS:
        val = row.get(letter, None)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        s = str(val).strip()
        if s and s.lower() != "nan":
            opts.append((letter, s))
    return opts


def format_prompt(question: str, options) -> str:
    q = str(question).strip()
    if options:
        opt_str = "\n".join(f"({l}) {t}" for l, t in options)
        return (
            f"Question: {q}\n"
            f"Options:\n{opt_str}\n"
            f"Answer with only the option letter."
        )
    # Options already in the question text.
    return f"{q}\n\nAnswer with only the option letter (A, B, C, or D)."


def parse_choice(response: str, valid_letters) -> str:
    """Find the LAST occurrence of a valid letter in the response (the model
    may reason first and state the answer at the end). Look at standalone
    letters first to avoid grabbing letters inside words."""
    valid = set(valid_letters)
    if not valid:
        return ""
    text = response.strip().upper()
    # Prefer standalone letters (word boundaries).
    import re as _re
    pattern = r"\b([" + "".join(sorted(valid)) + r"])\b"
    matches = _re.findall(pattern, text)
    if matches:
        return matches[-1]
    # Fallback: any matching letter, last occurrence.
    for ch in reversed(text):
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
                    help="Path to MMStar.tsv (VLMEvalKit distribution).")
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
                # MMStar is always 4-choice; if the TSV doesn't break out the
                # options into A..D columns, still allow A-D for parsing.
                valid_letters = [l for l, _ in options] if options else list(LETTERS)
                gold = str(row.get("answer", "")).strip().upper()
                image = _decode_image_b64(row["image"])

                prompt = format_prompt(row["question"], options)
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
                raw, pred, gold, valid_letters, err = "", "", "", [], repr(e)

            fout.write(json.dumps({
                "index": row.get("index"),
                "pred": pred,
                "gold": gold,
                "correct": (pred != "" and pred == gold),
                "raw": raw,
                "category": row.get("category"),
                "l2_category": row.get("l2_category"),
                "n_options": len(valid_letters),
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

    by_l1 = defaultdict(lambda: [0, 0])
    by_l2 = defaultdict(lambda: [0, 0])
    for r in all_results:
        k1 = r.get("category") or "_unknown"
        by_l1[k1][0] += int(r["correct"])
        by_l1[k1][1] += 1
        k2 = r.get("l2_category") or "_unknown"
        by_l2[k2][0] += int(r["correct"])
        by_l2[k2][1] += 1

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
        "by_category": fmt(by_l1),
        "by_l2_category": fmt(by_l2),
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)

    print("\n========== MMStar Eval ==========")
    print(f"Model:      {args.model_path}")
    print(f"TSV:        {args.tsv}")
    print(f"Total:      {n}")
    print(f"Correct:    {correct}")
    print(f"Accuracy:   {correct/n*100:.2f}%")
    print(f"Parse fail: {parse_fail}")
    print(f"Errors:     {errors}")
    print("\nBy L1 category:")
    for k, v in summary["by_category"].items():
        print(f"  {k:28s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
