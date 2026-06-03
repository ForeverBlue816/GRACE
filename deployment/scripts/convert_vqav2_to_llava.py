import argparse
import json
import os


VQAV2_PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."

# split name -> (questions filename, annotations filename, image folder name)
SPLIT_FILES = {
    "train": (
        "v2_OpenEnded_mscoco_train2014_questions.json",
        "v2_mscoco_train2014_annotations.json",
        "train2014",
    ),
    "val": (
        "v2_OpenEnded_mscoco_val2014_questions.json",
        "v2_mscoco_val2014_annotations.json",
        "val2014",
    ),
}


def image_filename(image_id, split_folder):
    # COCO convention: COCO_<split>_000000XXXXXX.jpg with 12-digit zero pad.
    return f"COCO_{split_folder}_{image_id:012d}.jpg"


def convert_split(base_dir, split_name, image_prefix):
    q_file, a_file, split_folder = SPLIT_FILES[split_name]
    with open(os.path.join(base_dir, q_file)) as f:
        questions = json.load(f)["questions"]
    with open(os.path.join(base_dir, a_file)) as f:
        annotations = json.load(f)["annotations"]

    # Join on question_id.
    ann_by_qid = {a["question_id"]: a for a in annotations}

    out, skipped = [], 0
    for q in questions:
        qid = q["question_id"]
        ann = ann_by_qid.get(qid)
        if ann is None:
            skipped += 1
            continue
        answer = ann.get("multiple_choice_answer", "").strip()
        question_text = q.get("question", "").strip()
        if not (answer and question_text):
            skipped += 1
            continue

        image_name = image_filename(q["image_id"], split_folder)
        image_path = os.path.join(image_prefix, "images", split_folder, image_name) \
            if image_prefix else os.path.join("images", split_folder, image_name)

        out.append({
            "id": str(qid),
            "split": split_name,
            "image": image_path,
            "conversations": [
                {"from": "human",
                 "value": f"{question_text}{VQAV2_PROMPT_SUFFIX}\n<image>"},
                {"from": "gpt", "value": answer},
            ],
        })
    return out, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert VQAv2 train+val into a merged LLaVA-format JSON."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Dir containing the v2_*.json files and images/ folder")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=list(SPLIT_FILES.keys()))
    parser.add_argument("--image_prefix", default="VQAv2",
                        help="Prefix prepended to image paths "
                             "(e.g. 'VQAv2' -> 'VQAv2/images/train2014/<file>.jpg'). "
                             "Pass '' to disable.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write the JSON (default: this script's dir)")
    parser.add_argument("--output_name", default=None,
                        help="Output filename (default: llava_vqav2_<splits>.json)")
    args = parser.parse_args()

    merged, total_skipped = [], 0
    for split in args.splits:
        items, skipped = convert_split(args.base_dir, split, args.image_prefix)
        print(f"  {split}: kept {len(items)}, skipped {skipped}")
        merged.extend(items)
        total_skipped += skipped

    print(f"Total: {len(merged)} samples (skipped {total_skipped})")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    if args.output_name is None:
        tag = "-".join(args.splits)
        output_name = f"llava_vqav2_{tag}.json"
    else:
        output_name = args.output_name
    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
