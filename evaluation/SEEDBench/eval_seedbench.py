"""SEED-Bench (image split) evaluation for a fine-tuned Qwen3-VL checkpoint.

Reads the official SEED-Bench distribution (`SEED-Bench.json` + an unzipped
image directory) rather than the VLMEvalKit TSV. Filters to entries with
`data_type == "image"` (SEED-Bench-IMG, ~14k items across 9 dimensions); the
video split is ignored.

Official JSON shape:
  {
    "questions": [
      {"question_id": "v1-101", "question": "...", "choice_a": ...,
       "choice_b": ..., "choice_c": ..., "choice_d": ..., "answer": "B",
       "data_id": "1454426.jpg", "data_type": "image",
       "question_type_id": 1},
      ...
    ],
    "question_type": {"Scene Understanding": 1, ...}
  }

Single-process or sharded over GPUs via torchrun. Each rank writes a JSONL
shard; rank 0 merges and prints metrics at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

LETTERS = "ABCD"


def load_data(json_path: str):
    with open(json_path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        items = d.get("questions", [])
        qtype_map = d.get("question_type", {})
    else:
        items = d
        qtype_map = {}
    # id -> name
    id2type = {int(v): k for k, v in qtype_map.items()}
    return items, id2type


def collect_options(row):
    opts = []
    for letter in LETTERS:
        v = row.get(f"choice_{letter.lower()}")
        if v is None or str(v).strip() == "":
            continue
        opts.append((letter, str(v)))
    return opts


def format_prompt(question: str, options) -> str:
    opt_str = "\n".join(f"({l}) {t}" for l, t in options)
    return (
        f"Question: {str(question).strip()}\n"
        f"Options:\n{opt_str}\n"
        f"Answer with only the option letter."
    )


def parse_choice(response: str, valid_letters) -> str:
    valid = set(valid_letters)
    for ch in response.strip().upper():
        if ch in valid:
            return ch
    return ""


def resolve_image_path(row, image_dir: str):
    data_id = row.get("data_id")
    if not data_id:
        return None
    # data_id may include extension; try as-is, then a few common suffixes.
    candidates = [data_id]
    if "." not in os.path.basename(data_id):
        candidates += [f"{data_id}.jpg", f"{data_id}.png", f"{data_id}.jpeg"]
    for c in candidates:
        p = os.path.join(image_dir, c)
        if os.path.isfile(p):
            return p
    return None


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
    ap.add_argument("--json_path", required=True,
                    help="Path to SEED-Bench.json.")
    ap.add_argument("--image_dir", required=True,
                    help="Directory holding image files (after unzipping "
                         "SEED-Bench-image.zip).")
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

    items, id2type = load_data(args.json_path)
    items = [it for it in items if str(it.get("data_type", "")).lower() == "image"]
    if args.limit > 0:
        items = items[: args.limit]
    shard = items[rank::world_size]
    if rank == 0:
        print(f"[data] image-only={len(items)} shard={len(shard)}", flush=True)

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

                img_path = resolve_image_path(row, args.image_dir)
                if img_path is None:
                    raise FileNotFoundError(f"image not found: {row.get('data_id')}")
                image = Image.open(img_path).convert("RGB")

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

            qtid = row.get("question_type_id")
            try:
                qtid_int = int(qtid) if qtid is not None else None
            except (TypeError, ValueError):
                qtid_int = None
            fout.write(json.dumps({
                "question_id": row.get("question_id"),
                "data_id": row.get("data_id"),
                "pred": pred,
                "gold": gold,
                "correct": (pred != "" and pred == gold),
                "raw": raw,
                "question_type_id": qtid_int,
                "question_type": id2type.get(qtid_int) if qtid_int is not None else None,
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

    by_type = defaultdict(lambda: [0, 0])
    for r in all_results:
        k = r.get("question_type") or f"type_{r.get('question_type_id')}"
        by_type[k][0] += int(r["correct"])
        by_type[k][1] += 1

    def fmt(d):
        return {k: {"acc": (c / n_), "correct": c, "total": n_}
                for k, (c, n_) in sorted(d.items())}

    summary = {
        "model_path": args.model_path,
        "json_path": args.json_path,
        "image_dir": args.image_dir,
        "total": n,
        "correct": correct,
        "overall_acc": correct / n if n else 0.0,
        "parse_fail": parse_fail,
        "errors": errors,
        "by_question_type": fmt(by_type),
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)

    print("\n========== SEED-Bench-IMG Eval ==========")
    print(f"Model:      {args.model_path}")
    print(f"Total:      {n}")
    print(f"Correct:    {correct}")
    print(f"Accuracy:   {correct/n*100:.2f}%")
    print(f"Parse fail: {parse_fail}")
    print(f"Errors:     {errors}")
    print("\nBy question type:")
    for k, v in summary["by_question_type"].items():
        print(f"  {k:30s}: {v['correct']:5d}/{v['total']:5d} = {v['acc']*100:.2f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
