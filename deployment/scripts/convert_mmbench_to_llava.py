import argparse
import base64
import csv
import io
import json
import os
import sys

# MMBench's base64 image cells are long — lift csv's default field cap.
csv.field_size_limit(sys.maxsize)

MMBENCH_PROMPT_SUFFIX = "Answer with the option's letter from the given choices directly."

# (filename, language tag) — newer 20231003 TSVs only; 20230712 is superseded.
DEFAULT_TSVS = [
    ("mmbench_dev_en_20231003.tsv", "en"),
    ("mmbench_dev_cn_20231003.tsv", "cn"),
]

OPTION_KEYS = ["A", "B", "C", "D"]


def is_blank(v):
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in ("", "nan", "none")


def decode_image_to_disk(b64_str, out_path):
    if os.path.exists(out_path):
        return True
    try:
        raw = base64.b64decode(b64_str)
    except Exception:
        return False
    with open(out_path, "wb") as f:
        f.write(raw)
    return True


def build_prompt(question, hint, options):
    """Reproduce LLaVA's MMBench eval prompt structure."""
    parts = []
    if not is_blank(hint):
        parts.append(f"Hint: {hint.strip()}")
    parts.append(f"Question: {question.strip()}")
    parts.append("Options:")
    for letter, text in options:
        parts.append(f"{letter}. {text}")
    parts.append(MMBENCH_PROMPT_SUFFIX)
    return "\n".join(parts)


def convert_tsv(tsv_path, lang, images_out_dir, image_prefix):
    out, skipped_no_answer, skipped_bad_image = 0, 0, 0
    samples = []

    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            index = row.get("index")
            question = row.get("question", "")
            hint = row.get("hint", "")
            answer = row.get("answer", "")
            image_b64 = row.get("image", "")

            if is_blank(index) or is_blank(question) or is_blank(answer):
                skipped_no_answer += 1
                continue

            options = []
            for key in OPTION_KEYS:
                val = row.get(key, "")
                if not is_blank(val):
                    options.append((key, str(val).strip()))
            if not options:
                skipped_no_answer += 1
                continue

            image_name = f"{index}.jpg"
            image_disk_path = os.path.join(images_out_dir, image_name)
            if not is_blank(image_b64):
                if not decode_image_to_disk(image_b64, image_disk_path):
                    skipped_bad_image += 1
                    continue
            elif not os.path.exists(image_disk_path):
                # Other language's TSV may have written it already; otherwise skip.
                skipped_bad_image += 1
                continue

            image_field = os.path.join(image_prefix, "images", image_name) \
                if image_prefix else os.path.join("images", image_name)

            prompt = build_prompt(question, hint, options)
            samples.append({
                "id": f"mmbench_{lang}_{index}",
                "language": lang,
                "category": row.get("category", ""),
                "image": image_field,
                "conversations": [
                    {"from": "human", "value": f"{prompt}\n<image>"},
                    {"from": "gpt", "value": str(answer).strip()},
                ],
            })
            out += 1

    return samples, out, skipped_no_answer, skipped_bad_image


def main():
    parser = argparse.ArgumentParser(
        description="Convert MMBench dev TSVs into one LLaVA-format JSON (images decoded to disk)."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Dir containing the mmbench_dev_*.tsv files")
    parser.add_argument("--tsvs", nargs="+", default=None,
                        help="Optional explicit list of TSV filenames to process. "
                             "Default: en+cn 20231003. Language tag is parsed from the filename.")
    parser.add_argument("--image_prefix", default="mmbench",
                        help="Prefix prepended to image paths "
                             "(e.g. 'mmbench' -> 'mmbench/images/<idx>.jpg'). Pass '' to disable.")
    parser.add_argument("--images_out_dir", default=None,
                        help="Where to write decoded .jpg images. "
                             "Default: <base_dir>/images/")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write the JSON (default: this script's dir)")
    parser.add_argument("--output_name", default="llava_mmbench_dev.json")
    args = parser.parse_args()

    images_out_dir = args.images_out_dir or os.path.join(args.base_dir, "images")
    os.makedirs(images_out_dir, exist_ok=True)

    if args.tsvs:
        tsv_list = []
        for name in args.tsvs:
            lang = "cn" if "_cn_" in name else ("en" if "_en_" in name else "unk")
            tsv_list.append((name, lang))
    else:
        tsv_list = DEFAULT_TSVS

    merged = []
    for fname, lang in tsv_list:
        path = os.path.join(args.base_dir, fname)
        if not os.path.exists(path):
            print(f"[warn] {path} not found, skipping")
            continue
        samples, kept, sk_ans, sk_img = convert_tsv(
            path, lang, images_out_dir, args.image_prefix
        )
        print(f"  {fname}: kept={kept}  no-answer/option={sk_ans}  bad-image={sk_img}")
        merged.extend(samples)

    print(f"Total: {len(merged)} samples")
    print(f"Decoded images written under: {images_out_dir}")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, args.output_name)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
