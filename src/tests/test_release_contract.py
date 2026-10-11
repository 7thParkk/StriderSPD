"""CPU-only release checks; these do not claim model inference validation."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.arguments import parse_args
from utils.dataset_manifest import load_manifest, validate_dataset
from utils.result_metrics import binary_metrics
from main import main


class ReleaseContractTests(unittest.TestCase):
    def test_eval_has_public_backbone_and_benchmark_defaults(self):
        args = parse_args(["--do_test", "--checkpoint_path", "paper.pth"])
        self.assertEqual(args.pretrained_llm_path, "Qwen/Qwen3-8B")
        self.assertEqual(args.llm_architecture_type, "qwen3")
        self.assertEqual(args.expected_eval_samples, 1720)
        self.assertEqual(Path(args.ass_path).name, "CFG")
        self.assertEqual(Path(args.pse_path).name, "pseudo")

    def test_joint_training_requires_its_initializer(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--do_train"])
        args = parse_args(["--do_train", "--pretrained_llm_path", "/merged"])
        self.assertEqual(args.gradient_accumulation_steps, 1)

    def test_missing_or_invalid_device_roles_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--do_test", "--checkpoint_path", "x", "--gnn_devices", "cpu"])
        args = parse_args(["--do_test", "--checkpoint_path", "x", "--gnn_devices",
                           "cuda:0,cuda:0,cuda:0"])
        self.assertEqual(args.gnn_devices, ["cuda:0"] * 3)

    @staticmethod
    def fixture(root):
        item = {"id": "sample", "project_dir": "project", "optimize_level": "O0",
                "a_ass_pse_file": "before.json", "b_ass_pse_file": "after.json",
                "category": "security"}
        (root / "benchmark_new_infos.json").write_text(json.dumps([item]), encoding="utf-8")
        for representation in ("CFG", "pseudo"):
            directory = root / representation / "project" / "O0"
            directory.mkdir(parents=True)
            for filename in ("before.json", "after.json"):
                (directory / filename).write_text("{}", encoding="utf-8")
        return item

    def test_missing_representation_fails_instead_of_skipping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / "pseudo/project/O0/after.json").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_dataset(root, root / "CFG", root / "pseudo", "benchmark_new")

    def test_count_and_duplicate_ids_fail_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = self.fixture(root)
            with self.assertRaises(ValueError):
                load_manifest(root, "benchmark_new", expected_count=1720)
            (root / "benchmark_new_infos.json").write_text(json.dumps([item, item]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(root, "benchmark_new")

    def test_preflight_runs_without_loading_ml_modules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            checkpoint = root / "placeholder.pth"
            checkpoint.write_bytes(b"preflight checks presence only")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(["--do_test", "--preflight", "--root_path", str(root),
                      "--checkpoint_path", str(checkpoint), "--expected_eval_samples", "1"])
            self.assertIn("PREFLIGHT_OK", output.getvalue())
            self.assertNotIn("torch", sys.modules)

    def test_paper_confusion_matrix_metrics(self):
        # Counts independently established by the archived 1720-record export.
        labels = [1] * 1010 + [0] * 710
        predictions = [1] * 967 + [0] * 43 + [1] * 208 + [0] * 502
        result = binary_metrics(labels, predictions)
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]),
                         (967, 502, 208, 43))
        for field, value in {
            "accuracy": 0.8540697674418605, "precision": 0.8229787234042554,
            "recall": 0.9574257425742574, "f1": 0.8851258581235698,
            "fpr": 0.29295770521722464,
        }.items():
            self.assertAlmostEqual(result[field], value, places=12)


if __name__ == "__main__":
    unittest.main()
