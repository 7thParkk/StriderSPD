"""Validate manifest alignment and representation availability before model loading."""

import json
from pathlib import Path


def load_manifest(root_path, split, expected_count=None):
    path = Path(root_path) / f"{split}_infos.json"
    with path.open(encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a nonempty list of samples")
    if expected_count is not None and len(items) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} samples, got {len(items)}")
    seen = set()
    required = ("id", "project_dir", "optimize_level", "a_ass_pse_file", "b_ass_pse_file", "category")
    for index, item in enumerate(items):
        missing = [field for field in required if field not in item]
        if missing:
            raise ValueError(f"{path}: row {index} lacks {missing}")
        identifier = str(item["id"])
        if identifier in seen:
            raise ValueError(f"{path}: duplicate sample ID {identifier}")
        seen.add(identifier)
        if item["category"] not in ("security", "non-security"):
            raise ValueError(f"{path}: row {index} has unknown category {item['category']!r}")
    return items


def representation_paths(item, ass_path, pse_path):
    relative = Path(item["project_dir"]) / item["optimize_level"]
    return tuple(Path(root) / relative / item[field]
                 for field in ("a_ass_pse_file", "b_ass_pse_file")
                 for root in (ass_path, pse_path))


def validate_dataset(root_path, ass_path, pse_path, split, expected_count=None):
    items = load_manifest(root_path, split, expected_count)
    missing = [(item["id"], str(path)) for item in items
               for path in representation_paths(item, ass_path, pse_path)
               if not path.is_file()]
    if missing:
        preview = "\n".join(f"sample {identifier}: {path}" for identifier, path in missing[:10])
        raise FileNotFoundError(f"{len(missing)} missing representation file(s):\n{preview}")
    return items
