"""Binary metrics matching the archived paper-checkpoint prediction export."""


def binary_metrics(labels, predictions, logits=None):
    if not labels or len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same nonzero length")
    if any(value not in (0, 1) for value in (*labels, *predictions)):
        raise ValueError("binary labels and predictions must be 0 or 1")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    result = {
        "num_samples": len(labels), "accuracy": (tp + tn) / len(labels),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fpr": fp / (fp + tn + 0.0001),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "pred_pos_rate": sum(predictions) / len(predictions),
        "label_pos_rate": sum(labels) / len(labels),
    }
    if logits is not None:
        if len(logits) != len(labels):
            raise ValueError("logit count does not match labels")
        result["mean_logit"] = sum(logits) / len(logits)
    return result
