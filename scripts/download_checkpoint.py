#!/usr/bin/env python3
"""Download the paper checkpoint with Hugging Face Hub and verify its SHA256."""
import argparse
import hashlib
from pathlib import Path

REPO_ID = "hanjin996/StriderSPD-V1"
FILENAME = "model_epoch5_best_acc_qwen3_8b_824.pth"
SHA256 = "991045a6bd0c8a30b113b1d2f633ed6d464f48ec06f4325b6adfef27db5e01e7"
SIZE_BYTES = 25682301190


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                                revision=args.revision, local_dir=str(args.output_dir)))
    if path.stat().st_size != SIZE_BYTES:
        raise SystemExit(f"Unexpected checkpoint size: {path.stat().st_size}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise SystemExit("Checkpoint SHA256 mismatch")
    print(f"Verified checkpoint: {path.resolve()}")


if __name__ == "__main__":
    main()
