"""HallusionBench evaluation for a fine-tuned Qwen3-VL checkpoint.

Reads the `lmms-lab/HallusionBench` parquet distribution:
    <data_dir>/image-*.parquet      (951 image-grounded yes/no questions)
    <data_dir>/non_image-*.parquet  (178 text-only yes/no questions)

Per-row fields: category, subcategory, visual_input, set_id, figure_id,
sample_note, question_id, question, gt_answer_details, gt_answer, filename,
image (HF Image -> dict {'bytes': ..., 'path': ...} after parquet load).
`gt_answer` is "1" (Yes) or "0" (No).

Three official-style metrics, rule-based (no GPT judge):
  - aAcc: per-question accuracy
  - qAcc: a (category, subcategory, set_id, question_id) group is correct iff
          every question in it is correct (image-vs-no-image consistency)
  - fAcc: a (category, subcategory, set_id, figure_id) group is correct iff
          every question on that figure is correct (text-only entries skipped)

Yes/no parsing scans for the first hit among a small keyword list.

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

YES_WORDS = {"yes", "true", "correct", "right", "is", "are", "does", "can"}
NO_WORDS = {"no", "not", "false", "incorrect", "wrong", "cannot",
            "isn't", "aren't", "doesn't", "don't"}

YESNO_RE = re.compile(r"\b([a-z']+)\b", re.IGNORECASE)


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
            f"No parquet files in {data_dir}. "
            f"Expected image-*.parquet and non_image-*.parquet."
        )
    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        df["__source"] = os.path.basename(fp)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def format_prompt(question: str) -> str:
    return f"{str(question).strip()}\nAnswer with only 'Yes' or 'No'."


def parse_yesno(response: str) -> str:
    """Returns '1' (yes), '0' (no), or '' (unparseable)."""
    text = response.strip().lower()
    if not text:
        return ""
    for m in YESNO_RE.finditer(text):
        w = m.group(1)
        if w in YES_WORDS:
            return "1"
        if w in NO_WORDS:
            return "0"
    return ""


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
                    help="Dir containing image-*.parquet and non_image-*.parquet.")
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
        for row in tqdm(shard, disable=(rank != 0), desc=f"rank{rank}"):
            try:
                image = _decode_image(row.get("image"))
                # visual_input "1" means image-grounded; but trust the actual
                # decoded image (some rows might be mis-flagged).
                prompt = format_prompt(str(row["question"]))
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
                pred = parse_yesno(raw)
                err = None
            except Exception as e:
                raw, pred, image, err = "", "", None, repr(e)

            gold = str(row.get("gt_answer", "")).strip()
            fout.write(json.dumps({
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "set_id": row.get("set_id"),
                "figure_id": row.get("figure_id"),
                "question_id": row.get("question_id"),
                "visual_input": str(row.get("visual_input", "0")),
                "has_image": image is not None,
                "pred": pred,
                "gold": gold,
                "correct": (pred != "" and pred == gold),
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
    parse_fail = sum(1 for r in all_results if r["pred"] == "" and r["error"] is None)
    errors = sum(1 for r in all_results if r["error"] is not None)

    q_groups = defaultdict(list)
    f_groups = defaultdict(list)
    by_cat = defaultdict(lambda: [0, 0])
    by_subcat = defaultdict(lambda: [0, 0])
    by_mod = defaultdict(lambda: [0, 0])
    for r in all_results:
        c = r.get("category") or "_unknown"
        sc = r.get("subcategory") or "_unknown"
        sid = r.get("set_id")
        fid = r.get("figure_id")
        qid = r.get("question_id")
        by_cat[c][0] += int(r["correct"]); by_cat[c][1] += 1
        by_subcat[f"{c}/{sc}"][0] += int(r["correct"]); by_subcat[f"{c}/{sc}"][1] += 1
        m = "image" if r["has_image"] else "text"
        by_mod[m][0] += int(r["correct"]); by_mod[m][1] += 1

        q_groups[(c, sc, sid, qid)].append(r["correct"])
        if r["has_image"]:
            f_groups[(c, sc, sid, fid)].append(r["correct"])

    qAcc_total = len(q_groups)
    qAcc_correct = sum(1 for v in q_groups.values() if all(v))
    fAcc_total = len(f_groups)
    fAcc_correct = sum(1 for v in f_groups.values() if all(v))

    def fmt(d):
        return {k: {"acc": (c / n_), "correct": c, "total": n_}
                for k, (c, n_) in sorted(d.items())}

    summary = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "total": n,
        "correct": correct,
        "aAcc": correct / n if n else 0.0,
        "qAcc": qAcc_correct / qAcc_total if qAcc_total else 0.0,
        "qAcc_correct_groups": qAcc_correct,
        "qAcc_total_groups": qAcc_total,
        "fAcc": fAcc_correct / fAcc_total if fAcc_total else 0.0,
        "fAcc_correct_groups": fAcc_correct,
        "fAcc_total_groups": fAcc_total,
        "parse_fail": parse_fail,
        "errors": errors,
        "by_modality": fmt(by_mod),
        "by_category": fmt(by_cat),
        "by_subcategory": fmt(by_subcat),
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)

    print("\n========== HallusionBench Eval ==========")
    print(f"Model:      {args.model_path}")
    print(f"Total:      {n}")
    print(f"aAcc:       {summary['aAcc']*100:.2f}%  ({correct}/{n})")
    print(f"qAcc:       {summary['qAcc']*100:.2f}%  ({qAcc_correct}/{qAcc_total})")
    print(f"fAcc:       {summary['fAcc']*100:.2f}%  ({fAcc_correct}/{fAcc_total})")
    print(f"Parse fail: {parse_fail}")
    print(f"Errors:     {errors}")
    print("\nBy modality:")
    for k, v in summary["by_modality"].items():
        print(f"  {k:6s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print("\nBy category:")
    for k, v in summary["by_category"].items():
        print(f"  {k:8s}: {v['correct']:4d}/{v['total']:4d} = {v['acc']*100:.2f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
