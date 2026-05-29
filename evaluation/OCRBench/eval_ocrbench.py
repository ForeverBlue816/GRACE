"""OCRBench evaluation for a fine-tuned Qwen3-VL checkpoint.

OCRBench (Liu et al., 2024) is a 1000-sample multimodal OCR benchmark with 5
task families:
  - Text Recognition
  - Scene Text-centric VQA
  - Doc-oriented VQA
  - Key Information Extraction
  - Handwritten Mathematical Expression Recognition (HME)

Reads the `echo840/OCRBench` HF parquet distribution:
    <data_dir>/test-*.parquet
Per-row fields: dataset, question, question_type, answer (list[str]),
image (HF Image -> dict {'bytes': ..., 'path': ...} after parquet load).

Official scoring is rule-based:
  - For most tasks: a sample is correct iff any gold answer (case-insensitive,
    whitespace-collapsed) appears as a substring of the model's response.
  - For HME (Handwritten Mathematical Expression Recognition): strict
    whitespace-stripped exact match (substring is too lenient for LaTeX).

The benchmark's final "OCRBench score" is the *sum of correct counts* across
the 5 task families (max 1000), which equals the total correct on a per-row
basis — we report both forms for clarity.

Single-process or sharded over GPUs via torchrun. Each rank writes a JSONL
shard; rank 0 merges and prints metrics at the end.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from collections import defaultdict
from glob import glob

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

HME_TYPE_KEYWORDS = ("handwritten mathematical", "hme")


def _decode_image(value):
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        b = value.get("bytes")
        if b:
            return Image.open(io.BytesIO(b)).convert("RGB")
        return None
    if isinstance(value, (bytes, bytearray)) and len(value) > 0:
        return Image.open(io.BytesIO(value)).convert("RGB")
    return None


def load_data(data_dir: str) -> pd.DataFrame:
    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files in {data_dir}. Expected test-*.parquet."
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def get_answers(row):
    a = row.get("answer")
    if a is None:
        a = row.get("answers")
    if a is None:
        return []
    if isinstance(a, str):
        return [a]
    try:
        return [str(x) for x in a]
    except TypeError:
        return [str(a)]


def is_hme(row) -> bool:
    t = str(row.get("question_type") or row.get("dataset") or "").lower()
    return any(k in t for k in HME_TYPE_KEYWORDS)


def _norm(s: str) -> str:
    s = s.lower().strip()
    return re.sub(r"\s+", " ", s)


def score_default(response: str, golds) -> bool:
    r = _norm(response)
    if not r:
        return False
    for g in golds:
        gn = _norm(str(g))
        if gn and gn in r:
            return True
    return False


def score_hme(response: str, golds) -> bool:
    r = re.sub(r"\s+", "", response.strip())
    if not r:
        return False
    for g in golds:
        gn = re.sub(r"\s+", "", str(g).strip())
        if gn and gn == r:
            return True
    return False


def format_prompt(question: str) -> str:
    # Use the OCRBench question verbatim. Each question already specifies its
    # own expected answer format (e.g., HME asks "using LaTeX format", VQA
    # tasks ask short answers). Appending a generic "single word or phrase"
    # hint conflicts with the LaTeX/structured-output cue on HME tasks and
    # tanks that subset (matches behavior of the official OCRBench eval).
    return str(question).strip()


def build_inputs(processor, image, prompt_text, device):
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
    ap.add_argument("--data_dir", required=True,
                    help="Dir containing OCRBench test-*.parquet.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="Bumped from 64 to fit long HME LaTeX expressions.")
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

    df = load_data(args.data_dir)
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
        for i, row in enumerate(tqdm(shard, disable=(rank != 0), desc=f"rank{rank}")):
            try:
                image = _decode_image(row.get("image"))
                if image is None:
                    raise ValueError("missing image")
                prompt = format_prompt(row["question"])
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
                golds = get_answers(row)
                correct = score_hme(raw, golds) if is_hme(row) else score_default(raw, golds)
                err = None
            except Exception as e:
                raw, correct, golds, err = "", False, get_answers(row), repr(e)

            fout.write(json.dumps({
                "idx": i * world_size + rank,
                "dataset": row.get("dataset"),
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "golds": golds,
                "raw": raw,
                "correct": bool(correct),
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
    errors = sum(1 for r in all_results if r["error"] is not None)

    by_type = defaultdict(lambda: [0, 0])
    by_ds = defaultdict(lambda: [0, 0])
    for r in all_results:
        t = r.get("question_type") or "_unknown"
        by_type[t][0] += int(r["correct"]); by_type[t][1] += 1
        d = r.get("dataset") or "_unknown"
        by_ds[d][0] += int(r["correct"]); by_ds[d][1] += 1

    def fmt(d):
        return {k: {"acc": (c / n_), "correct": c, "total": n_}
                for k, (c, n_) in sorted(d.items())}

    ocrbench_score = sum(c for c, _ in by_type.values())

    summary = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "total": n,
        "correct": correct,
        "overall_acc": correct / n if n else 0.0,
        "ocrbench_score": ocrbench_score,
        "errors": errors,
        "by_question_type": fmt(by_type),
        "by_dataset": fmt(by_ds),
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)

    print("\n========== OCRBench Eval ==========")
    print(f"Model:           {args.model_path}")
    print(f"Total:           {n}")
    print(f"Correct:         {correct}")
    print(f"Accuracy:        {correct/n*100:.2f}%")
    print(f"OCRBench score:  {ocrbench_score} / {n}")
    print(f"Errors:          {errors}")
    print("\nBy question_type:")
    for k, v in summary["by_question_type"].items():
        print(f"  {k:42s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print("\nBy dataset:")
    for k, v in summary["by_dataset"].items():
        print(f"  {k:42s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
