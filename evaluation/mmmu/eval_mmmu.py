"""MMMU evaluation (validation split, local) for a fine-tuned Qwen3-VL checkpoint.

Reads the HuggingFace-format MMMU dataset laid out as one parquet per subject
per split (e.g. <data_dir>/Math/validation-00000-of-00001.parquet). Each row
carries `id`, `question`, `options` (a string repr of a list), `answer`,
`question_type` ("multiple-choice" | "open"), `subfield`, and up to seven
inline image columns `image_1` ... `image_7` (any unused ones are null).

The question text references images positionally as `<image 1>`, `<image 2>`,
etc. We collect every referenced image in order and feed them to the model
alongside the prompt; remaining image columns are ignored.

Scoring:
  - multiple-choice: extract first valid option letter from the response.
  - open: case-insensitive normalized match against the gold answer
          (also tries number extraction when the gold parses as a float).
    This is a simple rule-based proxy for the official GPT-judge eval, so
    open-ended numbers will roughly match but free-text recall will be
    slightly under-counted. Use the gpt-judge path in evaluation/mmmu/ if
    you need leaderboard-comparable open scores.

Single-process or sharded over GPUs via torchrun. Each rank writes a JSONL
shard; rank 0 merges and prints metrics at the end.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import string
import time
from collections import defaultdict
from glob import glob

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

LETTERS = "ABCDEFGHIJ"
MAX_IMAGES = 7
IMAGE_TAG_RE = re.compile(r"<image\s+(\d+)\s*>", re.IGNORECASE)


def _decode_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value and value["bytes"] is not None:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise ValueError(f"Unrecognized image cell type: {type(value)}")


def _is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return value.get("bytes") is not None
    if isinstance(value, (bytes, bytearray)) and len(value) > 0:
        return True
    if isinstance(value, Image.Image):
        return True
    return False


def _parse_options(opts_field):
    if isinstance(opts_field, list):
        return [str(o) for o in opts_field]
    if isinstance(opts_field, str):
        s = opts_field.strip()
        if not s or s == "[]":
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(o) for o in parsed]
        except (ValueError, SyntaxError):
            pass
    return []


def load_data(data_dir: str, split: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*", f"{split}-*.parquet")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No parquet files match {pattern}. "
            f"Expected layout: {data_dir}/<Subject>/{split}-*.parquet"
        )
    dfs = []
    for fp in files:
        subject = os.path.basename(os.path.dirname(fp))
        df = pd.read_parquet(fp)
        df["__subject"] = subject
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def collect_images(row):
    """Return images in the order they're first referenced by <image N> in
    question text. Images referenced but missing (None column) are skipped."""
    text = str(row.get("question", ""))
    order = []
    seen = set()
    for m in IMAGE_TAG_RE.finditer(text):
        idx = int(m.group(1))
        if idx in seen:
            continue
        seen.add(idx)
        col = f"image_{idx}"
        val = row.get(col)
        if _is_present(val):
            order.append(_decode_image(val))
    # Fallback: if the question never said <image N> but image_1 exists, use it.
    if not order and _is_present(row.get("image_1")):
        order.append(_decode_image(row["image_1"]))
    return order


def _strip_image_tags(text: str) -> str:
    return IMAGE_TAG_RE.sub("<image>", str(text))


def format_prompt_mc(question, options) -> str:
    q = _strip_image_tags(question).strip()
    opt_str = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    return (
        f"Question: {q}\n"
        f"Options:\n{opt_str}\n"
        f"Answer with only the option letter."
    )


def format_prompt_open(question) -> str:
    q = _strip_image_tags(question).strip()
    return (
        f"{q}\n"
        f"Answer with a short phrase or number only."
    )


def parse_choice(response: str, n_choices: int) -> str:
    valid = set(LETTERS[:n_choices])
    for ch in response.strip().upper():
        if ch in valid:
            return ch
    return ""


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _try_number(s: str):
    try:
        return float(s.strip().replace(",", "").rstrip("."))
    except ValueError:
        return None


def score_open(response: str, gold: str) -> bool:
    raw = response.strip()
    if not raw or not gold:
        return False
    g_num = _try_number(gold)
    if g_num is not None:
        m = re.search(r"-?\d+(?:\.\d+)?", raw)
        if m:
            r_num = _try_number(m.group(0))
            if r_num is not None and abs(r_num - g_num) < 1e-4:
                return True
    r_norm = _normalize(raw)
    g_norm = _normalize(gold)
    if not g_norm:
        return False
    return g_norm == r_norm or g_norm in r_norm.split()


def build_inputs(processor, images, prompt_text, device):
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs)
    return inputs.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_dir", required=True,
                    help="MMMU root containing per-subject folders with parquet files.")
    ap.add_argument("--split", default="validation",
                    choices=["validation", "dev", "test"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=32,
                    help="Used for open-ended; multiple-choice forces 8.")
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

    df = load_data(args.data_dir, args.split)
    if args.limit > 0:
        df = df.head(args.limit)
    rows = df.to_dict(orient="records")
    shard = rows[rank::world_size]
    if rank == 0:
        print(f"[data] split={args.split} total={len(rows)} shard={len(shard)}",
              flush=True)

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    shard_path = f"{args.output}.rank{rank}.jsonl"

    t0 = time.time()
    with open(shard_path, "w") as fout:
        for row in tqdm(shard, disable=(rank != 0), desc=f"rank{rank}"):
            qtype = str(row.get("question_type") or "multiple-choice").lower()
            gold_raw = row.get("answer", "")
            try:
                images = collect_images(row)
                if qtype.startswith("multi"):
                    options = _parse_options(row.get("options"))
                    if not options:
                        raise ValueError("multi-choice row has no options")
                    prompt = format_prompt_mc(row["question"], options)
                    max_toks = 8
                else:
                    options = []
                    prompt = format_prompt_open(row["question"])
                    max_toks = args.max_new_tokens

                inputs = build_inputs(processor, images, prompt, device)
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_toks,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                gen = out[0][inputs["input_ids"].shape[1]:]
                raw = processor.tokenizer.decode(gen, skip_special_tokens=True)

                if qtype.startswith("multi"):
                    pred = parse_choice(raw, len(options))
                    gold = str(gold_raw).strip().upper()
                    correct = (pred != "" and pred == gold)
                else:
                    pred = raw.strip()
                    gold = str(gold_raw).strip()
                    correct = score_open(pred, gold)
                err = None
            except Exception as e:
                raw, pred, gold, correct, err = "", "", str(gold_raw), False, repr(e)

            fout.write(json.dumps({
                "id": row.get("id"),
                "subject": row.get("__subject"),
                "subfield": row.get("subfield"),
                "question_type": qtype,
                "n_images": len(collect_images(row)) if err is None else 0,
                "pred": pred,
                "gold": gold,
                "correct": bool(correct),
                "raw": raw,
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

    by_subj = defaultdict(lambda: [0, 0])
    by_type = defaultdict(lambda: [0, 0])
    for r in all_results:
        k = r.get("subject") or "_unknown"
        by_subj[k][0] += int(r["correct"])
        by_subj[k][1] += 1
        t = r.get("question_type") or "_unknown"
        by_type[t][0] += int(r["correct"])
        by_type[t][1] += 1

    def fmt(d):
        return {k: {"acc": (c / n_), "correct": c, "total": n_}
                for k, (c, n_) in sorted(d.items())}

    summary = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "split": args.split,
        "total": n,
        "correct": correct,
        "overall_acc": correct / n if n else 0.0,
        "errors": errors,
        "by_subject": fmt(by_subj),
        "by_question_type": fmt(by_type),
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)

    print("\n========== MMMU Eval ==========")
    print(f"Model:    {args.model_path}")
    print(f"Split:    {args.split}")
    print(f"Total:    {n}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {correct/n*100:.2f}%")
    print(f"Errors:   {errors}")
    print("\nBy question type:")
    for k, v in summary["by_question_type"].items():
        print(f"  {k:18s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print("\nBy subject:")
    for k, v in summary["by_subject"].items():
        print(f"  {k:34s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
