"""MathVista (testmini) evaluation for a fine-tuned Qwen3-VL checkpoint.

Reads the AI4Math/MathVista HF parquet distribution. Default split is
`testmini` (1000 problems with gold answers); `test` (5141) has no public
gold and is only useful for CodaLab leaderboard submission.

Protocol follows the official eval (https://github.com/lupantech/MathVista):
  - Prompt is the pre-formatted `query` field from the parquet (already
    contains question + lettered options + "Hint: ... at the end").
    Falls back to a hand-built prompt if `query` is empty.
  - `max_new_tokens` is set high (256 by default) because the official Hint
    encourages the model to reason and place the final letter/number "at
    the end" of the response.
  - Answer extraction is the LAST valid letter (multi-choice) or LAST
    number (free-form numeric) in the response.
  - `precision` is interpreted as the *number of decimal places* (per the
    official codebase): both predicted and gold are rounded to that many
    decimal places before comparing.
  - For free-form text/list, fall back to case-insensitive normalized
    substring match.

This is a rule-based proxy for the official answer-extraction + GPT-judge
pipeline. Numerical scores track closely; verbose free-form text answers
may be 1-2 points below the GPT-judge version.

Single-process or sharded over GPUs via torchrun. Each rank writes a JSONL
shard; rank 0 merges and prints metrics at the end.
"""

from __future__ import annotations

import argparse
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
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _is_none_like(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "none", "null"):
        return True
    return False


def _decode_image(row):
    v = row.get("decoded_image")
    if isinstance(v, Image.Image):
        return v.convert("RGB")
    if isinstance(v, dict):
        b = v.get("bytes")
        if b:
            return Image.open(io.BytesIO(b)).convert("RGB")
        p = v.get("path")
        if p and os.path.isfile(p):
            return Image.open(p).convert("RGB")
    if isinstance(v, (bytes, bytearray)) and v:
        return Image.open(io.BytesIO(v)).convert("RGB")
    return None


def load_data(data_dir: str, split: str) -> pd.DataFrame:
    pats = [
        os.path.join(data_dir, f"{split}-*.parquet"),
        os.path.join(data_dir, "data", f"{split}-*.parquet"),
    ]
    for p in pats:
        files = sorted(glob(p))
        if files:
            return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raise FileNotFoundError(
        f"No parquet files match {split}-*.parquet under {data_dir} (or {data_dir}/data)."
    )


def _to_choice_list(v):
    if _is_none_like(v):
        return None
    if isinstance(v, (list, tuple)):
        lst = list(v)
    else:
        try:
            lst = list(v)
        except TypeError:
            return None
    lst = [str(c) for c in lst if not _is_none_like(c)]
    return lst or None


def fallback_prompt(row, choices) -> str:
    """Construct a prompt mirroring the official `query` field when missing."""
    q = str(row["question"]).strip()
    parts = [f"Question: {q}"]
    if choices:
        opt = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))
        parts.append(f"\nChoices:\n{opt}")
        parts.append("\nHint: Please answer the question and provide the correct "
                     "option letter, e.g., A, B, C, D, at the end.")
    else:
        atype = str(row.get("answer_type") or "").lower()
        if atype == "integer":
            hint = ("Hint: Please answer the question requiring an integer answer "
                    "and provide the final value, e.g., 1, 2, 3, at the end.")
        elif atype == "float":
            prec = row.get("precision")
            try:
                n = int(prec) if prec is not None else 2
            except (TypeError, ValueError):
                n = 2
            hint = (f"Hint: Please answer the question requiring a floating-point "
                    f"number with {n} decimal place(s) and provide the final value, "
                    f"e.g., 1.{'2'*n}, 3.{'4'*n}, at the end.")
        else:
            hint = ("Hint: Please answer the question and provide the final "
                    "answer at the end.")
        parts.append("\n" + hint)
    return "\n".join(parts)


def get_prompt(row, choices, use_query: bool) -> str:
    if use_query:
        q = row.get("query")
        if not _is_none_like(q):
            return str(q)
    return fallback_prompt(row, choices)


def parse_choice_last(response: str, n_choices: int) -> int:
    """Return the LAST valid option-letter index in the response (or -1)."""
    valid = set(LETTERS[:n_choices])
    last = -1
    for ch in response.upper():
        if ch in valid:
            last = LETTERS.index(ch)
    return last


def extract_last_number(response: str):
    matches = NUM_RE.findall(response.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _round_to(x: float, n_decimals):
    try:
        n = int(n_decimals)
    except (TypeError, ValueError):
        n = 4
    if n < 0:
        n = 4
    return round(float(x), n)


def _norm_text(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def score_free_form(response: str, gold, precision, answer_type: str) -> bool:
    raw = response.strip()
    if not raw or _is_none_like(gold):
        return False

    is_numeric = answer_type in ("integer", "float") or _is_none_like(answer_type)
    if is_numeric:
        try:
            g_num = float(str(gold).replace(",", "").strip())
        except ValueError:
            g_num = None
        if g_num is not None:
            r_num = extract_last_number(raw)
            if r_num is None:
                return False
            n_dec = precision
            if _is_none_like(n_dec):
                # default to a few digits; integers will round-trip exactly
                n_dec = 4
            return _round_to(r_num, n_dec) == _round_to(g_num, n_dec)
        # fall through to text comparison

    r_norm = _norm_text(raw)
    g_norm = _norm_text(str(gold))
    if not g_norm:
        return False
    return g_norm == r_norm or g_norm in r_norm


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
                    help="MathVista root (containing data/<split>-*.parquet "
                         "or directly <split>-*.parquet).")
    ap.add_argument("--split", default="testmini", choices=["testmini", "test"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="High default because Hint asks for answer 'at the end' "
                         "after free-form reasoning.")
    ap.add_argument("--no_official_query", action="store_true",
                    help="Don't use the pre-built `query` field; build prompt ourselves.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])
    args = ap.parse_args()
    use_query = not args.no_official_query

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[init] world_size={world_size} model={args.model_path} "
              f"use_official_query={use_query}", flush=True)

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
            qtype = str(row.get("question_type") or "").lower()
            answer_type = str(row.get("answer_type") or "").lower()
            gold = row.get("answer")
            choices = _to_choice_list(row.get("choices"))
            try:
                image = _decode_image(row)
                prompt = get_prompt(row, choices, use_query)

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

                if qtype == "multi_choice" and choices:
                    pred_idx = parse_choice_last(raw, len(choices))
                    pred = choices[pred_idx] if pred_idx >= 0 else ""
                    correct = (pred != "" and str(pred).strip() == str(gold).strip())
                else:
                    pred = raw.strip()
                    correct = score_free_form(
                        pred, gold, row.get("precision"), answer_type
                    )
                err = None
            except Exception as e:
                raw, pred, correct, err = "", "", False, repr(e)

            meta = row.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            skills = meta.get("skills") or []
            try:
                skills = list(skills)
            except TypeError:
                skills = []
            fout.write(json.dumps({
                "pid": row.get("pid"),
                "question_type": qtype,
                "answer_type": answer_type,
                "task": meta.get("task"),
                "category": meta.get("category"),
                "skills": [str(s) for s in skills],
                "grade": meta.get("grade"),
                "source": meta.get("source"),
                "pred": str(pred),
                "gold": str(gold),
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

    by_qtype = defaultdict(lambda: [0, 0])
    by_task = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    by_skill = defaultdict(lambda: [0, 0])
    for r in all_results:
        q = r.get("question_type") or "_unknown"
        by_qtype[q][0] += int(r["correct"]); by_qtype[q][1] += 1
        t = r.get("task") or "_unknown"
        by_task[t][0] += int(r["correct"]); by_task[t][1] += 1
        c = r.get("category") or "_unknown"
        by_cat[c][0] += int(r["correct"]); by_cat[c][1] += 1
        # skills is a list -> a sample contributes to every skill bucket it tags
        for s in (r.get("skills") or []):
            by_skill[s][0] += int(r["correct"]); by_skill[s][1] += 1

    def fmt(d):
        return {k: {"acc": (c / n_), "correct": c, "total": n_}
                for k, (c, n_) in sorted(d.items())}

    summary = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "split": args.split,
        "use_official_query": use_query,
        "total": n,
        "correct": correct,
        "overall_acc": correct / n if n else 0.0,
        "errors": errors,
        "by_question_type": fmt(by_qtype),
        "by_task": fmt(by_task),
        "by_category": fmt(by_cat),
        "by_skill": fmt(by_skill),
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)

    print("\n========== MathVista Eval ==========")
    print(f"Model:    {args.model_path}")
    print(f"Split:    {args.split}  (use_official_query={use_query})")
    print(f"Total:    {n}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {correct/n*100:.2f}%")
    print(f"Errors:   {errors}")
    print("\nBy question_type:")
    for k, v in summary["by_question_type"].items():
        print(f"  {k:14s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print("\nBy task:")
    for k, v in summary["by_task"].items():
        print(f"  {k:34s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print("\nBy skill:")
    for k, v in summary["by_skill"].items():
        print(f"  {k:30s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
