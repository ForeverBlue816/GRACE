"""ScienceQA evaluation for a fine-tuned Qwen3-VL checkpoint.

Standard ScienceQA multi-choice protocol: prompt the model with the question,
optional textual hint, and lettered options; parse the first valid option
letter from the response; report accuracy overall, by subject, and by modality
(image vs text-only questions).

Single-process or sharded over GPUs via torchrun. Each rank writes a JSONL
shard; rank 0 merges and prints metrics at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

LETTERS = "ABCDEFGHIJ"


def load_data(problems_path: str, splits_path: str, split: str):
    with open(problems_path) as f:
        problems = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)
    pids = splits[split]
    return [{"pid": pid, **problems[pid]} for pid in pids]


def format_prompt(item) -> str:
    choices = item["choices"]
    opts = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))
    parts = []
    hint = item.get("hint") or ""
    if hint.strip():
        parts.append(f"Context: {hint.strip()}")
    parts.append(f"Question: {item['question'].strip()}")
    parts.append(f"Options:\n{opts}")
    parts.append("Answer with only the option letter.")
    return "\n".join(parts)


def parse_choice(response: str, n_choices: int) -> int:
    valid = set(LETTERS[:n_choices])
    for ch in response.strip().upper():
        if ch in valid:
            return LETTERS.index(ch)
    return -1


def resolve_image_path(item, split: str, image_root: str) -> str | None:
    img_field = item.get("image")
    if not img_field:
        return None
    p = os.path.join(image_root, split, item["pid"], img_field)
    return p if os.path.isfile(p) else None


def build_inputs(processor, image, prompt_text, device, dtype):
    content = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
    if image is not None:
        kwargs["images"] = [image]
    inputs = processor(**kwargs)
    return inputs.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--image_root", required=True,
                    help="Parent dir of train/val/test image folders.")
    ap.add_argument("--output", required=True,
                    help="Path to write merged JSON results (rank 0).")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only evaluate the first N samples (debugging).")
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
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    items = load_data(args.problems, args.splits, args.split)
    if args.limit > 0:
        items = items[: args.limit]
    shard = items[rank::world_size]
    if rank == 0:
        print(f"[data] total={len(items)} shard={len(shard)} (split={args.split})",
              flush=True)

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    shard_path = f"{args.output}.rank{rank}.jsonl"

    t0 = time.time()
    results = []
    with open(shard_path, "w") as fout:
        for it in tqdm(shard, disable=(rank != 0), desc=f"rank{rank}"):
            img_path = resolve_image_path(it, args.split, args.image_root)
            image = Image.open(img_path).convert("RGB") if img_path else None

            prompt = format_prompt(it)
            try:
                inputs = build_inputs(processor, image, prompt, device, torch.bfloat16)
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                gen = out[0][inputs["input_ids"].shape[1]:]
                raw = processor.tokenizer.decode(gen, skip_special_tokens=True)
                pred = parse_choice(raw, len(it["choices"]))
                err = None
            except Exception as e:
                raw, pred, err = "", -1, repr(e)

            row = {
                "pid": it["pid"],
                "pred": pred,
                "gold": it["answer"],
                "correct": pred == it["answer"],
                "raw": raw,
                "subject": it.get("subject"),
                "topic": it.get("topic"),
                "category": it.get("category"),
                "has_image": image is not None,
                "n_choices": len(it["choices"]),
                "error": err,
            }
            results.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    if rank == 0:
        print(f"[rank0 done] elapsed={(time.time()-t0)/60:.1f} min", flush=True)

    # Sync before rank0 merges shards.
    if is_dist:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()

    if rank != 0:
        return

    all_results = []
    for r in range(world_size):
        sp = f"{args.output}.rank{r}.jsonl"
        with open(sp) as f:
            for line in f:
                all_results.append(json.loads(line))

    n = len(all_results)
    n_correct = sum(1 for r in all_results if r["correct"])
    n_parse_fail = sum(1 for r in all_results if r["pred"] == -1)

    by_subj = defaultdict(lambda: [0, 0])
    by_mod = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    for r in all_results:
        by_subj[r["subject"] or "_unknown"][0] += int(r["correct"])
        by_subj[r["subject"] or "_unknown"][1] += 1
        k = "image" if r["has_image"] else "text"
        by_mod[k][0] += int(r["correct"])
        by_mod[k][1] += 1
        by_cat[r["category"] or "_unknown"][0] += int(r["correct"])
        by_cat[r["category"] or "_unknown"][1] += 1

    def fmt(d):
        return {k: {"acc": (c / n_), "correct": c, "total": n_}
                for k, (c, n_) in sorted(d.items())}

    summary = {
        "model_path": args.model_path,
        "split": args.split,
        "total": n,
        "correct": n_correct,
        "overall_acc": n_correct / n if n else 0.0,
        "parse_fail": n_parse_fail,
        "by_subject": fmt(by_subj),
        "by_modality": fmt(by_mod),
        "by_category": fmt(by_cat),
    }

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)

    print("\n========== ScienceQA Eval ==========")
    print(f"Model:      {args.model_path}")
    print(f"Split:      {args.split}")
    print(f"Total:      {n}")
    print(f"Correct:    {n_correct}")
    print(f"Accuracy:   {n_correct/n*100:.2f}%")
    print(f"Parse fail: {n_parse_fail}")
    print("\nBy modality:")
    for k, v in summary["by_modality"].items():
        print(f"  {k:6s}: {v['correct']:5d}/{v['total']:5d} = {v['acc']*100:.2f}%")
    print("\nBy subject:")
    for k, v in summary["by_subject"].items():
        print(f"  {k:20s}: {v['correct']:5d}/{v['total']:5d} = {v['acc']*100:.2f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
