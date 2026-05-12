"""Run a ProtectAI prompt-injection classifier baseline on ShieldAgent JSON data.

This is a same-split text-classifier baseline for public stress-test datasets.
It scans the concatenated untrusted tool-response text from each sample. It is
not a full replacement for ShieldAgent because it does not evaluate tool-chain
anomaly detection, intent verification, or sandbox policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


def _load_samples(path: Path) -> list[dict[str, Any]]:
    records = []
    for file in sorted(path.glob("*.json")):
        records.append(json.loads(file.read_text()))
    return records


def _tool_response_text(record: dict[str, Any]) -> str:
    responses = record.get("tool_responses", [])
    if responses:
        return "\n\n".join(str(item.get("response", "")) for item in responses)
    return str(record.get("prompt", ""))


def run_baseline(attack_path: Path, benign_path: Path, output: Path, model_id: str) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "This baseline requires transformers and torch. Run with: "
            "uv run --with transformers --with torch python scripts/run_protectai_baseline.py"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    attacks = _load_samples(attack_path)
    benign = _load_samples(benign_path)
    samples = [(record, 1) for record in attacks] + [(record, 0) for record in benign]

    tp = fp = tn = fn = 0
    latencies = []

    with torch.no_grad():
        for record, label in samples:
            text = _tool_response_text(record)
            start = time.perf_counter()
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            logits = model(**inputs).logits
            pred = int(torch.argmax(logits, dim=-1).item())
            latencies.append((time.perf_counter() - start) * 1000)

            if label == 1 and pred == 1:
                tp += 1
            elif label == 1 and pred == 0:
                fn += 1
            elif label == 0 and pred == 1:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(samples) if samples else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0

    result = {
        "model": model_id,
        "attack_path": str(attack_path),
        "benign_path": str(benign_path),
        "num_attack_samples": len(attacks),
        "num_benign_samples": len(benign),
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "benign": {
            "false_positive_rate": false_positive_rate,
            "false_positives": fp,
            "total": fp + tn,
        },
        "latency": {
            "mean_ms": mean(latencies) if latencies else 0.0,
            "num_samples": len(latencies),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacks", type=Path, required=True)
    parser.add_argument("--benign", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/protectai_baseline.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    result = run_baseline(args.attacks, args.benign, args.output, args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
