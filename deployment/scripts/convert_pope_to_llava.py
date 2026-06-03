import argparse
import json
import os


POPE_PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."

# POPE only ships eval splits — all yes/no labeled — under coco/.
POPE_FILES = ["coco_pope_random.json", "coco_pope_popular.json", "coco_pope_adversarial.json"]


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def convert_file(path, subset_name, image_prefix, image_folder):
    entries = load_jsonl(path)
    out, skipped = [], 0
    for e in entries:
        image = e.get("image")
        question = (e.get("text") or "").strip()
        label = (e.get("label") or "").strip()
        if not (image and question and label):
            skipped += 1
            continue

        image_path = os.path.join(image_prefix, image_folder, image) \
            if image_prefix else os.path.join(image_folder, image)

        out.append({
            "id": f"pope_{subset_name}_{e.get('question_id')}",
            "subset": subset_name,
            "image": image_path,
            "conversations": [
                {"from": "human",
                 "value": f"{question}{POPE_PROMPT_SUFFIX}\n<image>"},
                {"from": "gpt", "value": label},
            ],
        })
    return out, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert POPE (random/popular/adversarial) into one LLaVA-format JSON."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Dir containing coco/coco_pope_*.json and val2014/ image folder")
    parser.add_argument("--image_folder", default="val2014",
                        help="Subfolder under image_prefix where images live (default: val2014)")
    parser.add_argument("--image_prefix", default="pope",
                        help="Prefix prepended to image paths "
                             "(e.g. 'pope' -> 'pope/val2014/<file>.jpg'). Pass '' to disable.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write the JSON (default: this script's dir)")
    parser.add_argument("--output_name", default="llava_pope_all.json")
    args = parser.parse_args()

    merged, total_skipped = [], 0
    for fname in POPE_FILES:
        path = os.path.join(args.base_dir, "coco", fname)
        if not os.path.exists(path):
            print(f"[warn] {path} not found, skipping")
            continue
        subset = fname.replace("coco_pope_", "").replace(".json", "")
        items, skipped = convert_file(path, subset, args.image_prefix, args.image_folder)
        print(f"  {subset}: kept {len(items)}, skipped {skipped}")
        merged.extend(items)
        total_skipped += skipped

    print(f"Total: {len(merged)} samples (skipped {total_skipped})")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, args.output_name)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
