"""AI2D evaluation for a fine-tuned Qwen3-VL checkpoint.

AI2D is a 4-way multiple-choice diagram understanding benchmark; every sample
has an image. Data is read directly from HuggingFace-style parquet shards
(e.g. data/test-00000-of-00002.parquet). Rule-based letter extraction is
sufficient — no GPT judge needed.

Single-process or sharded over GPUs via torchrun. Each rank writes a JSONL
shard; rank 0 merges and prints metrics at the end.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from collections import defaultdict
from glob import glob

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

LETTERS = "ABCD"


def _decode_image(value) -> Image.Image:
    """Parquet image column may be a PIL Image, bytes, or {'bytes': ...} dict."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise ValueError(f"Unrecognized image cell type: {type(value)}")


def _normalize_answer(ans, n_choices: int) -> int:
    if isinstance(ans, (int,)) or (isinstance(ans, str) and ans.isdigit()):
        idx = int(ans)
        return idx if 0 <= idx < n_choices else -1
    if isinstance(ans, str) and len(ans.strip()) == 1:
        ch = ans.strip().upper()
        if ch in LETTERS[:n_choices]:
            return LETTERS.index(ch)
    return -1


def load_data(data_dir: str):
    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # Column name normalization (lmms-lab/ai2d uses these; tolerate variants)
    rename = {}
    if "question" not in df.columns and "Question" in df.columns:
        rename["Question"] = "question"
    if "options" not in df.columns and "choices" in df.columns:
        rename["choices"] = "options"
    if rename:
        df = df.rename(columns=rename)
    return df


def format_prompt(question: str, options) -> str:
    opts = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(options))
    return (
        f"Question: {str(question).strip()}\n"
        f"Options:\n{opts}\n"
        f"Answer with only the option letter."
    )


def parse_choice(response: str, n_choices: int) -> int:
    valid = set(LETTERS[:n_choices])
    for ch in response.strip().upper():
        if ch in valid:
            return LETTERS.index(ch)
    return -1


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
    ap.add_argument("--data_dir", required=True,
                    help="Dir with AI2D parquet files (e.g. .../AI2D/data).")
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
        for i, row in enumerate(tqdm(shard, disable=(rank != 0), desc=f"rank{rank}")):
            try:
                question = row["question"]
                options = list(row["options"])
                gold = _normalize_answer(row["answer"], len(options))
                image = _decode_image(row["image"])

                prompt = format_prompt(question, options)
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
                pred = parse_choice(raw, len(options))
                err = None
            except Exception as e:
                raw, pred, gold, err = "", -1, -1, repr(e)

            fout.write(json.dumps({
                "idx": int(row.get("question_id", row.get("index", i * world_size + rank))),
                "pred": pred,
                "gold": gold,
                "correct": pred == gold and gold != -1,
                "raw": raw,
                "n_choices": len(row.get("options", [])) if isinstance(row.get("options"), (list, tuple)) else 4,
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
    parse_fail = sum(1 for r in all_results if r["pred"] == -1 and r["error"] is None)
    errors = sum(1 for r in all_results if r["error"] is not None)

    summary = {
        "model_path": args.model_path,
        "total": n,
        "correct": correct,
        "overall_acc": correct / n if n else 0.0,
        "parse_fail": parse_fail,
        "errors": errors,
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)

    print("\n========== AI2D Eval ==========")
    print(f"Model:      {args.model_path}")
    print(f"Total:      {n}")
    print(f"Correct:    {correct}")
    print(f"Accuracy:   {correct/n*100:.2f}%")
    print(f"Parse fail: {parse_fail}")
    print(f"Errors:     {errors}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
