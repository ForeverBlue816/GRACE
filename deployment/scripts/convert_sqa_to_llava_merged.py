import argparse
import json
import os
from convert_sqa_to_llava_base_prompt import build_prompt_chatbot


def convert_split(problems, split_indices, split_name, prompt_format, is_test,
                  image_prefix=""):
    split_problems = build_prompt_chatbot(
        problems, split_indices, prompt_format,
        use_caption=False, is_test=is_test)

    out = []
    for prob_id, (input_text, output_text) in split_problems.items():
        if input_text.startswith('Question: '):
            input_text = input_text.replace('Question: ', '')
        if output_text.startswith('Answer: '):
            output_text = output_text.replace('Answer: ', '')

        raw = problems[prob_id]
        if raw['image'] is None:
            out.append({
                "id": prob_id,
                "split": split_name,
                "conversations": [
                    {'from': 'human', 'value': f"{input_text}"},
                    {'from': 'gpt', 'value': f"{output_text}"},
                ],
            })
        else:
            out.append({
                "id": prob_id,
                "split": split_name,
                "image": os.path.join(image_prefix, split_name, prob_id, raw['image']) if image_prefix else os.path.join(split_name, prob_id, raw['image']),
                "conversations": [
                    {'from': 'human', 'value': f"{input_text}\n<image>"},
                    {'from': 'gpt', 'value': f"{output_text}"},
                ],
            })
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Merge ScienceQA train/val/test into one LLaVA-format JSON."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Dir containing problems.json and pid_splits.json")
    parser.add_argument("--prompt_format", default="QCM-LEA")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to include (default: all three)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write the JSON (default: this script's dir)")
    parser.add_argument("--output_name", default=None,
                        help="Output filename (default: llava_<splits>_<fmt>.json)")
    parser.add_argument("--image_prefix", default="ScienceQA",
                        help="Prefix prepended to every image path "
                             "(e.g. 'ScienceQA' -> 'ScienceQA/train/<pid>/<img>'). "
                             "Pass '' to disable.")
    args = parser.parse_args()

    with open(os.path.join(args.base_dir, "pid_splits.json")) as f:
        pid_splits = json.load(f)
    with open(os.path.join(args.base_dir, "problems.json")) as f:
        problems = json.load(f)

    merged = []
    for split in args.splits:
        if split not in pid_splits:
            print(f"[warn] split '{split}' not in pid_splits.json, skipping")
            continue
        items = convert_split(
            problems, pid_splits[split], split, args.prompt_format,
            is_test=False, image_prefix=args.image_prefix,
        )
        print(f"  {split}: {len(items)} samples")
        merged.extend(items)

    print(f"Total samples: {len(merged)}")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    if args.output_name is None:
        tag = "-".join(args.splits)
        output_name = f"llava_{tag}_{args.prompt_format}.json"
    else:
        output_name = args.output_name
    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
