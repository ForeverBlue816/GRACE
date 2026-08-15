import ast
import os
import pathlib
import sys
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DEPLOY_SCRIPT = ROOT / "qwen-vl-finetune" / "scripts" / "deploy_awq_qwen.py"
sys.path.insert(0, str(ROOT / "qwen-vl-finetune"))

from qwenvl.data import data_list


class ReleaseContractTests(unittest.TestCase):
    def test_primary_checkpoint_is_the_packed_awq_build(self):
        readme = README.read_text(encoding="utf-8")
        self.assertIn(
            "--load-packed ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ",
            readme,
        )
        self.assertIn("huggingface.co/spaces/ForeverBlue/GRACE-VLM", readme)

    def test_deploy_script_accepts_hub_repositories(self):
        source = DEPLOY_SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("resolve_model_dir", functions)
        self.assertIn("snapshot_download", source)

    def test_sharegpt4v_registry_uses_environment_root(self):
        with patch.dict(os.environ, {"SHAREGPT4V_ROOT": "/tmp/sharegpt4v"}):
            config = data_list(["sharegpt4v_mix665k%10"])[0]
        self.assertEqual(config["sampling_rate"], 0.1)
        self.assertEqual(
            pathlib.Path(config["data_path"]),
            pathlib.Path("/tmp/sharegpt4v/data").resolve(),
        )
        self.assertTrue(
            config["annotation_path"].endswith(
                "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"
            )
        )


if __name__ == "__main__":
    unittest.main()
