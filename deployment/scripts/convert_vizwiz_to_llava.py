import argparse
import json
import os
import re
from collections import Counter

# VizWiz image names look like "VizWiz_<split>_<id>.jpg" — the <split> portion
# tells us which folder it lives in, regardless of which json references it.
_IMAGE_FOLDER_RE = re.compile(r"^VizWiz_(train|val|test)_", re.IGNORECASE)


def image_folder_for(image_name, default_split):
    m = _IMAGE_FOLDER_RE.match(image_name)
    return m.group(1).lower() if m else default_split


VIZWIZ_PROMPT_SUFFIX = (
    "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\n"
    "Answer the question using a single word or phrase."
)


def pick_answer(entry):
    """Pull a single training label out of a VizWiz entry.

    Standard VizWiz: `answers` is a list of dicts with `answer` field — pick the
    mode (most common). Falls back to plain string answers or a single `answer`
    field for non-standard variants.
    """
    if "answers" in entry and entry["answers"]:
        items = entry["answers"]
        if isinstance(items[0], dict):
            answers = [a.get("answer", "").strip() for a in items if a.get("answer")]
        else:
            answers = [str(a).strip() for a in items if a]
        if not answers:
            return None
        return Counter(answers).most_common(1)[0][0]
    if "answer" in entry and entry["answer"]:
        return str(entry["answer"]).strip()
    return None


def convert_split(json_path, split_name, image_prefix):
    with open(json_path) as f:
        entries = json.load(f)

    out, skipped = [], 0
    for entry in entries:
        image = entry.get("image")
        question = entry.get("question", "").strip()
        answer = pick_answer(entry)
        if not (image and question and answer):
            skipped += 1
            continue

        folder = image_folder_for(image, split_name)
        image_path = os.path.join(image_prefix, folder, image) \
            if image_prefix else os.path.join(folder, image)
        sample_id = os.path.splitext(image)[0]

        out.append({
            "id": sample_id,
            "split": split_name,
            "image": image_path,
            "conversations": [
                {"from": "human", "value": f"{question}{VIZWIZ_PROMPT_SUFFIX}\n<image>"},
                {"from": "gpt", "value": answer},
            ],
        })
    return out, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert VizWiz train+val into a merged LLaVA-format JSON."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Dir containing train.json / val.json (and train/ val/ image folders)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--image_prefix", default="VizWiz",
                        help="Prefix prepended to image paths "
                             "(e.g. 'VizWiz' -> 'VizWiz/train/<img>.jpg'). Pass '' to disable.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write the JSON (default: this script's dir)")
    parser.add_argument("--output_name", default=None,
                        help="Output filename (default: llava_vizwiz_<splits>.json)")
    args = parser.parse_args()

    merged, total_skipped = [], 0
    for split in args.splits:
        json_path = os.path.join(args.base_dir, f"{split}.json")
        if not os.path.exists(json_path):
            print(f"[warn] {json_path} not found, skipping")
            continue
        items, skipped = convert_split(json_path, split, args.image_prefix)
        print(f"  {split}: kept {len(items)}, skipped {skipped}")
        merged.extend(items)
        total_skipped += skipped

    print(f"Total: {len(merged)} samples (skipped {total_skipped})")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    if args.output_name is None:
        tag = "-".join(args.splits)
        output_name = f"llava_vizwiz_{tag}.json"
    else:
        output_name = args.output_name
    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
