import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler
from accelerate import Accelerator

from config.logger import setup_logger
from utils.collect import custom_collate_fn
from utils.move_to_device import move_to_device
from utils.dataset_manifest import load_manifest
from utils.result_metrics import binary_metrics

logger = setup_logger()


def evaluate(model, eval_dataset, args, accelerator=None):
    """Evaluate in manifest order and export aligned paper-checkpoint predictions."""
    owns_accelerator = accelerator is None
    if accelerator is None:
        accelerator = Accelerator(mixed_precision="bf16", device_placement=False)
    if accelerator.num_processes != 1:
        raise RuntimeError("Run V1 with one Python process; its GPUs use model parallelism.")
    loader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=custom_collate_fn, sampler=SequentialSampler(eval_dataset))
    if owns_accelerator:
        model, loader = accelerator.prepare(model, loader)
    else:
        loader = accelerator.prepare(loader)

    exporting = bool(args.do_test)
    items = None
    output = None
    if exporting:
        items = load_manifest(args.root_path, args.eval_split,
                              args.expected_eval_samples or None)
        if len(eval_dataset) != len(items):
            raise RuntimeError("Dataset length differs from manifest; refusing unaligned export")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        partial_path = output_dir / "predictions.jsonl.partial"
        final_path = output_dir / "predictions.jsonl"
        output = partial_path.open("w", encoding="utf-8", newline="\n")

    model.eval()
    predictions, labels, scores_all = [], [], []
    cursor = 0
    try:
        for batch in loader:
            batch = move_to_device(batch, args.device)
            with torch.no_grad():
                with accelerator.autocast():
                    logits = model(batch)
                if exporting:
                    # Match the exact archived export: float32 CPU sigmoid >= 0.5.
                    scores = logits[:, 0].detach().float().cpu()
                    probabilities = torch.sigmoid(scores)
                    batch_predictions = (probabilities >= 0.5).int().tolist()
                else:
                    # Keep the V1 validation threshold/precision during training.
                    probabilities = torch.sigmoid(logits[:, 0])
                    batch_predictions = (probabilities >= 0.5).int().cpu().tolist()
                    scores = logits[:, 0].detach().float().cpu()
                batch_labels = batch["label"].int().cpu().tolist()
            if exporting:
                for offset, (prediction, label, score, probability) in enumerate(zip(
                        batch_predictions, batch_labels, scores.tolist(), probabilities.tolist())):
                    index = cursor + offset
                    if index >= len(items):
                        raise RuntimeError("Loader exported more rows than the manifest")
                    sample = items[index]
                    if label != int(sample["category"] == "security"):
                        raise RuntimeError(f"Label mismatch at manifest row {index}")
                    record = {
                        "index": index, "sample_id": sample["id"],
                        "prediction": prediction,
                        "prediction_label": "security" if prediction else "non-security",
                        "label": label, "label_name": sample["category"],
                        "correct": prediction == label, "logit": float(score),
                        "probability_security": float(probability), "sample": sample,
                    }
                    output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                output.flush()
            cursor += len(batch_predictions)
            predictions.extend(batch_predictions)
            labels.extend(batch_labels)
            scores_all.extend(scores.tolist())
            if exporting and (cursor % 100 == 0 or cursor == len(items)):
                print(f"exported={cursor}/{len(items)}", flush=True)
        if cursor != len(eval_dataset):
            raise RuntimeError(f"Expected {len(eval_dataset)} predictions, got {cursor}")
    finally:
        if output is not None:
            output.close()

    result = binary_metrics(labels, predictions, scores_all)
    if exporting:
        os.replace(partial_path, final_path)
        result.update({
            "prediction_file": str(final_path),
            "benchmark_manifest": str(Path(args.root_path) / f"{args.eval_split}_infos.json"),
            "checkpoint": str(Path(args.checkpoint_path)),
            "code_root": str(Path(__file__).resolve().parents[1]),
            "pretrained_llm_path": args.pretrained_llm_path,
            "unixcoder_path": args.unixcoder_path,
        })
        temporary_summary = output_dir / "summary.json.partial"
        with temporary_summary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_summary, output_dir / "summary.json")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    for key in ("accuracy", "precision", "recall", "f1", "fpr"):
        logger.info("%s = %s", key, result[key])
    print(f'eval_acc={result["accuracy"]:.4f}, eval_f1={result["f1"]:.4f}, '
          f'eval_precision={result["precision"]:.4f}, eval_fpr={result["fpr"]:.4f}')
    return result
