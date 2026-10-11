#!/usr/bin/env python3
"""Validate the released 1,720-sample benchmark using the Python standard library."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


EXPECTED_SAMPLES = 1720
EXPECTED_LABELS = {"security": 1010, "non-security": 710}
OPTIMIZATION_LEVELS = {"O0", "O1", "O2", "O3", "Os"}
MANIFEST_FIELDS = {
    "id", "optimize_level", "project_dir", "source", "commit_id", "src_file",
    "a_bin_file", "b_bin_file", "a_ass_pse_file", "b_ass_pse_file",
    "diff_functions", "category",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def component(value, description):
    require(isinstance(value, str) and value not in ("", ".", ".."),
            f"Invalid {description}: {value!r}")
    require("/" not in value and "\\" not in value and ":" not in value,
            f"{description} must be one relative path component: {value!r}")
    return value


def check_manifest(rows):
    require(isinstance(rows, list), "Manifest must contain a JSON list")
    require(len(rows) == EXPECTED_SAMPLES,
            f"Expected {EXPECTED_SAMPLES} samples, found {len(rows)}")
    ids, pairs = set(), set()
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"Row {index} is not an object")
        require(MANIFEST_FIELDS <= row.keys(),
                f"Row {index} lacks fields: {sorted(MANIFEST_FIELDS - row.keys())}")
        require(type(row["id"]) is int and row["id"] not in ids,
                f"Row {index} has an invalid or duplicate id")
        ids.add(row["id"])
        require(row["category"] in EXPECTED_LABELS, f"Invalid label at row {index}")
        require(row["optimize_level"] in OPTIMIZATION_LEVELS,
                f"Invalid optimization level at row {index}")
        component(row["project_dir"], f"row {index} project_dir")
        for field in ("source", "commit_id", "src_file", "a_bin_file", "b_bin_file"):
            require(isinstance(row[field], str) and row[field],
                    f"Row {index} has invalid {field}")
        require(isinstance(row["diff_functions"], list)
                and all(isinstance(value, str) for value in row["diff_functions"]),
                f"Row {index} has invalid diff_functions")
        for field in ("a_ass_pse_file", "b_ass_pse_file"):
            value = component(row[field], f"row {index} {field}")
            require(value.endswith(".json"), f"Row {index} has a non-JSON filename")
        require(row["a_ass_pse_file"] != row["b_ass_pse_file"],
                f"Row {index} has identical pre/post filenames")
        pair = tuple(row[field] for field in
                     ("project_dir", "optimize_level", "a_ass_pse_file", "b_ass_pse_file"))
        require(pair not in pairs, f"Duplicate sample pair at row {index}")
        pairs.add(pair)
    labels = dict(Counter(row["category"] for row in rows))
    require(labels == EXPECTED_LABELS, f"Unexpected label counts: {labels}")
    return labels


def check_modality(root, rows, modality):
    folder = root / modality
    require(folder.is_dir(), f"Missing modality directory: {folder}")
    referenced = {
        Path(row["project_dir"]) / row["optimize_level"] / row[f"{side}_ass_pse_file"]
        for row in rows for side in ("a", "b")
    }
    actual = {path.relative_to(folder) for path in folder.rglob("*.json")}
    require(referenced == actual,
            f"{modality}: {len(referenced - actual)} missing and "
            f"{len(actual - referenced)} unreferenced JSON files")
    require(len(actual) == EXPECTED_SAMPLES * 2,
            f"{modality}: expected {EXPECTED_SAMPLES * 2} distinct pre/post files")
    total_bytes = 0
    for relative in sorted(actual):
        path = folder / relative
        data = read_json(path)
        require(isinstance(data, dict), f"Not a JSON object: {path}")
        if modality == "CFG":
            for field in ("nodes", "edges", "boundary", "jmp_instruction"):
                require(isinstance(data.get(field), dict), f"Invalid {field}: {path}")
            require(all(isinstance(value, list)
                        and all(isinstance(instruction, str) for instruction in value)
                        for value in data["nodes"].values()), f"Invalid node instructions: {path}")
            require(all(isinstance(value, list) for value in data["edges"].values()),
                    f"Invalid adjacency lists: {path}")
        else:
            require(set(data) == {"pseudo_code"}
                    and isinstance(data["pseudo_code"], list)
                    and all(isinstance(line, str) for line in data["pseudo_code"]),
                    f"Invalid pseudo-code representation: {path}")
        total_bytes += path.stat().st_size
    return {"files": len(actual), "bytes": total_bytes}


def check_predictions(path, rows):
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    require(len(records) == len(rows), "Prediction and manifest row counts differ")
    counts = Counter()
    for index, (record, row) in enumerate(zip(records, rows)):
        label = int(row["category"] == "security")
        require(record.get("index") == index and record.get("sample_id") == row["id"],
                f"Prediction index or sample ID mismatch at row {index}")
        require(record.get("sample") == row,
                f"Prediction metadata differs from manifest at row {index}")
        require(record.get("label") == label and record.get("label_name") == row["category"],
                f"Prediction label mismatch at row {index}")
        prediction = record.get("prediction")
        require(type(prediction) is int and prediction in (0, 1),
                f"Invalid prediction at row {index}")
        require(record.get("prediction_label") == ("security" if prediction else "non-security"),
                f"Prediction name mismatch at row {index}")
        require(record.get("correct") is (prediction == label),
                f"Prediction correctness mismatch at row {index}")
        probability = record.get("probability_security")
        require(isinstance(probability, (int, float)) and math.isfinite(probability)
                and 0 <= probability <= 1, f"Invalid probability at row {index}")
        require(prediction == int(probability >= 0.5),
                f"Thresholded prediction mismatch at row {index}")
        counts[label, prediction] += 1
    tp, tn, fp, fn = (counts[1, 1], counts[0, 0], counts[0, 1], counts[1, 0])
    return {
        "num_samples": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(rows),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        # Preserve the original evaluator's 0.0001 denominator adjustment.
        "fpr": fp / (fp + tn + 0.0001),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "benchmark_1720")
    parser.add_argument("--predictions", type=Path,
                        help="Optional reference or newly exported prediction JSONL")
    parser.add_argument("--metrics", type=Path,
                        help="Optional metrics JSON to compare with --predictions")
    args = parser.parse_args()
    if args.metrics and not args.predictions:
        parser.error("--metrics requires --predictions")
    try:
        manifest = args.benchmark_dir / "benchmark_new_infos.json"
        rows = read_json(manifest)
        result = {
            "num_samples": len(rows), "label_counts": check_manifest(rows),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "project_counts": dict(Counter(row["source"] for row in rows)),
            "optimization_counts": dict(Counter(row["optimize_level"] for row in rows)),
            "modalities": {name: check_modality(args.benchmark_dir, rows, name)
                           for name in ("CFG", "pseudo")},
        }
        if args.predictions:
            result["prediction_metrics"] = check_predictions(args.predictions, rows)
            if args.metrics:
                expected = read_json(args.metrics)
                for field, value in result["prediction_metrics"].items():
                    require(field in expected
                            and math.isclose(value, expected[field], rel_tol=1e-12, abs_tol=1e-12),
                            f"Reference metric mismatch: {field}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Benchmark validation failed: {error}\n")


if __name__ == "__main__":
    main()
