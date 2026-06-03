import argparse
import json
import os


MME_PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."
MME_ROOT_NAME = "MME_Benchmark_release_version"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image_for(stem, search_dirs):
    """Look for <stem>.<ext> in each search_dirs. Returns the first match or None."""
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for ext in IMG_EXTS:
            cand = os.path.join(d, stem + ext)
            if os.path.exists(cand):
                return cand
    return None


def parse_qa_file(path):
    """Each line: '<question>\\t<answer>'. Returns list of (question, answer)."""
    qa = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            question = parts[0].strip()
            answer = parts[1].strip()
            if question and answer:
                qa.append((question, answer))
    return qa


def walk_category(category_dir):
    """Yield (txt_path, image_path) pairs for all questions in a category.

    Handles both MME layouts:
      A) flat:   <category>/<id>.txt + <category>/<id>.jpg
      B) split:  <category>/questions_answers_YN/<id>.txt + <category>/images/<id>.jpg
    """
    img_dirs = [category_dir, os.path.join(category_dir, "images")]
    txt_dirs = [category_dir, os.path.join(category_dir, "questions_answers_YN")]

    seen = set()
    for tdir in txt_dirs:
        if not os.path.isdir(tdir):
            continue
        for name in sorted(os.listdir(tdir)):
            if not name.endswith(".txt"):
                continue
            stem = os.path.splitext(name)[0]
            if stem in seen:
                continue
            seen.add(stem)
            txt_path = os.path.join(tdir, name)
            img_path = find_image_for(stem, img_dirs)
            yield txt_path, img_path, stem


def convert(base_dir, image_prefix):
    mme_root = os.path.join(base_dir, MME_ROOT_NAME)
    if not os.path.isdir(mme_root):
        raise SystemExit(f"Expected directory not found: {mme_root}")

    categories = sorted(
        d for d in os.listdir(mme_root)
        if os.path.isdir(os.path.join(mme_root, d))
    )

    merged = []
    stats = {}
    for cat in categories:
        cat_dir = os.path.join(mme_root, cat)
        kept, skipped_missing_img, skipped_bad_line = 0, 0, 0

        for txt_path, img_path, stem in walk_category(cat_dir):
            if img_path is None:
                skipped_missing_img += 1
                continue
            qa = parse_qa_file(txt_path)
            if not qa:
                skipped_bad_line += 1
                continue

            # Path inside the JSON is relative to --image_folder at training time.
            rel = os.path.relpath(img_path, base_dir)  # e.g. MME_Benchmark.../artwork/10002.jpg
            image_field = os.path.join(image_prefix, rel) if image_prefix else rel

            for i, (question, answer) in enumerate(qa):
                merged.append({
                    "id": f"mme_{cat}_{stem}_{i}",
                    "category": cat,
                    "image": image_field,
                    "conversations": [
                        {"from": "human",
                         "value": f"{question}{MME_PROMPT_SUFFIX}\n<image>"},
                        {"from": "gpt", "value": answer},
                    ],
                })
                kept += 1

        stats[cat] = (kept, skipped_missing_img, skipped_bad_line)
        print(f"  {cat:25s} kept={kept:5d}  no-image={skipped_missing_img:3d}  bad-txt={skipped_bad_line:3d}")

    print(f"Total: {len(merged)} samples across {len(categories)} categories")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Convert MME benchmark into one LLaVA-format JSON."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Dir containing MME_Benchmark_release_version/")
    parser.add_argument("--image_prefix", default="mme",
                        help="Prefix prepended to image paths "
                             "(e.g. 'mme' -> 'mme/MME_Benchmark.../artwork/<id>.jpg'). "
                             "Pass '' to disable.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write the JSON (default: this script's dir)")
    parser.add_argument("--output_name", default="llava_mme_all.json")
    args = parser.parse_args()

    merged = convert(args.base_dir, args.image_prefix)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, args.output_name)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
