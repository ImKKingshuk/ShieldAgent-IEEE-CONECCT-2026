"""Run deterministic adaptive-evasion stress tests against the sanitizer.

The stress harness mutates attack samples with common prompt-injection evasion
forms, then measures sanitizer-only detection. It is intended as a supplemental
robustness check, not as a replacement for an adaptive human red-team study.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from shieldagent.core.config import SanitizerConfig
from shieldagent.defense.sanitizer import ToolResponseSanitizer


TARGET_WORDS = (
    "ignore",
    "previous",
    "instructions",
    "reveal",
    "system",
    "prompt",
    "bypass",
    "security",
    "curl",
    "bash",
    "sudo",
    "base64",
)

HOMOGLYPHS = str.maketrans({
    "a": "а",
    "e": "е",
    "i": "і",
    "o": "о",
    "p": "р",
    "y": "у",
    "s": "ѕ",
})


def _load_samples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    return [json.loads(file.read_text()) for file in files]


def _replace_words(text: str, transform: Callable[[str], str]) -> str:
    import re

    updated = text
    for word in TARGET_WORDS:
        updated = re.sub(
            rf"\b{re.escape(word)}\b",
            lambda match: transform(match.group(0)),
            updated,
            flags=re.IGNORECASE,
        )
    return updated


def _space_letters(text: str) -> str:
    return _replace_words(text, lambda word: " ".join(word))


def _punctuate_letters(text: str) -> str:
    return _replace_words(text, lambda word: ".".join(word))


def _zero_width_letters(text: str) -> str:
    return _replace_words(text, lambda word: "\u200b".join(word))


def _homoglyph_words(text: str) -> str:
    return _replace_words(text, lambda word: word.translate(HOMOGLYPHS))


MUTATIONS: dict[str, Callable[[str], str]] = {
    "identity": lambda text: text,
    "spaced_letters": _space_letters,
    "punctuated_letters": _punctuate_letters,
    "zero_width_letters": _zero_width_letters,
    "homoglyph_words": _homoglyph_words,
}


def _sample_texts(record: dict[str, Any]) -> list[tuple[str, str]]:
    responses = record.get("tool_responses") or []
    if not responses:
        return [("prompt", str(record.get("prompt", "")))]
    return [
        (str(item.get("tool_name", "unknown")), str(item.get("response", "")))
        for item in responses
    ]


def _is_detected(record: dict[str, Any], sanitizer: ToolResponseSanitizer, mutation: Callable[[str], str]) -> tuple[bool, float]:
    elapsed_ms = 0.0
    for tool_name, text in _sample_texts(record):
        start = time.perf_counter()
        result = sanitizer.sanitize(mutation(text), tool_name)
        elapsed_ms += (time.perf_counter() - start) * 1000
        if not result.is_safe:
            return True, elapsed_ms
    return False, elapsed_ms


def run_stress(
    attack_path: Path,
    benign_path: Path | None,
    output: Path,
    limit: int | None,
) -> dict[str, Any]:
    sanitizer = ToolResponseSanitizer(SanitizerConfig(enabled=True, block_on_detection=True))
    attacks = _load_samples(attack_path, limit)
    benign = _load_samples(benign_path, limit) if benign_path else []

    mutation_results = {}
    identity_detected_records = []
    for record in attacks:
        detected, _ = _is_detected(record, sanitizer, MUTATIONS["identity"])
        identity_detected_records.append(detected)

    for name, mutation in MUTATIONS.items():
        detections = []
        latencies = []
        for record in attacks:
            detected, latency_ms = _is_detected(record, sanitizer, mutation)
            detections.append(detected)
            latencies.append(latency_ms)
        retained = [
            detected
            for detected, identity_detected in zip(detections, identity_detected_records, strict=True)
            if identity_detected
        ]
        mutation_results[name] = {
            "attack_samples": len(attacks),
            "detected": sum(detections),
            "missed": len(detections) - sum(detections),
            "detection_rate": sum(detections) / len(detections) if detections else 0.0,
            "identity_detected_subset": {
                "samples": len(retained),
                "detected_after_mutation": sum(retained),
                "retention_rate": sum(retained) / len(retained) if retained else 0.0,
            },
            "mean_latency_ms": mean(latencies) if latencies else 0.0,
        }

    benign_flags = []
    benign_latencies = []
    for record in benign:
        detected, latency_ms = _is_detected(record, sanitizer, MUTATIONS["identity"])
        benign_flags.append(detected)
        benign_latencies.append(latency_ms)

    result = {
        "attack_path": str(attack_path),
        "benign_path": str(benign_path) if benign_path else None,
        "scope": "sanitizer-only adaptive evasion stress test",
        "mutations": mutation_results,
        "benign_identity": {
            "benign_samples": len(benign),
            "false_positives": sum(benign_flags),
            "false_positive_rate": sum(benign_flags) / len(benign_flags) if benign_flags else 0.0,
            "mean_latency_ms": mean(benign_latencies) if benign_latencies else 0.0,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacks", type=Path, required=True)
    parser.add_argument("--benign", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_stress.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    result = run_stress(args.attacks, args.benign, args.output, args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
