"""Dataset registry for the released GRACE Qwen3-VL recipes.

The preprocessing pipeline is adapted from QwenLM/Qwen3-VL (Apache-2.0).
Set ``SHAREGPT4V_ROOT`` to the directory described in the project README before
starting training.
"""

import os
import re
from pathlib import Path


_SHAREGPT4V_FILES = {
    "sharegpt4v_mix665k": (
        "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"
    ),
    "sharegpt4v_instruct_100k": (
        "sharegpt4v_instruct_gpt4-vision_cap100k.json"
    ),
}


def _sharegpt4v_config(annotation_file):
    root_value = os.getenv("SHAREGPT4V_ROOT")
    if not root_value:
        raise EnvironmentError(
            "SHAREGPT4V_ROOT is not set. Point it at the ShareGPT4V directory "
            "containing the annotation JSON files and data/ image tree."
        )
    root = Path(root_value).expanduser().resolve()
    return {
        "annotation_path": str(root / annotation_file),
        "data_path": str(root / "data"),
    }


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    configs = []
    for requested_name in dataset_names:
        sampling_rate = parse_sampling_rate(requested_name)
        dataset_name = re.sub(r"%(\d+)$", "", requested_name)
        try:
            annotation_file = _SHAREGPT4V_FILES[dataset_name]
        except KeyError as exc:
            available = ", ".join(sorted(_SHAREGPT4V_FILES))
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
            ) from exc
        config = _sharegpt4v_config(annotation_file)
        config["sampling_rate"] = sampling_rate
        configs.append(config)
    return configs
