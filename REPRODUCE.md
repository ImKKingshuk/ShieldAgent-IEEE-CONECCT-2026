# Reproducing ShieldAgent

This document describes the reproducibility workflow for the public IEEE CONECCT 2026 artifact.

## Environment

- Python 3.12+
- `uv`
- Docker is optional

## Setup

```bash
git clone https://github.com/ImKKingshuk/ShieldAgent-IEEE-CONECCT-2026.git
cd ShieldAgent-IEEE-CONECCT-2026
uv sync
```

## Unit Tests

```bash
uv run pytest tests/ -v
```

## Generate a Representative Dataset

The generator creates four attack categories and benign samples. The command below creates a small deterministic sample set for validating the pipeline:

```bash
uv run python scripts/generate_attacks.py --output data/generated --samples 2 --benign 4 --seed 1043
```

For a paper-scale run matching the reported 1,000 attack and 500 benign ratio:

```bash
uv run python scripts/generate_attacks.py --output data/generated_full --samples 250 --benign 500 --seed 1043
```

## Run Benchmark

```bash
uv run python scripts/run_experiments.py benchmark --attacks data/generated/attacks --benign data/generated/benign
```

The artifact includes pretrained GNN anomaly-detector weights at
`models/shieldagent_gnn.pt`. If the file is present, `ShieldAgentConfig.from_env()`
loads it automatically unless `ANOMALY_MODEL_PATH` overrides the path.

For ablations:

```bash
uv run python scripts/run_experiments.py benchmark --attacks data/generated/attacks --benign data/generated/benign --ablation
```

## Supplemental Public-Benchmark Imports

These adapters convert public benchmark records into ShieldAgent's JSON sample
format for stress testing. They are supplemental checks, not direct replacements
for the paper's agentic tool-chain benchmark.

```bash
uv run --with agentdojo python scripts/import_agentdojo.py --output data/external/agentdojo
uv run python scripts/run_experiments.py benchmark --attacks data/external/agentdojo/attacks --benign data/external/agentdojo/benign

uv run --with datasets python scripts/import_shieldlm.py --output data/external/shieldlm
uv run python - <<'PY'
import asyncio
from pathlib import Path
from shieldagent.evaluation.benchmark import BenchmarkConfig, run_benchmark

config = BenchmarkConfig(
    attack_dataset_path=Path("data/external/shieldlm/attacks"),
    benign_dataset_path=Path("data/external/shieldlm/benign"),
    output_path=Path("results/shieldlm_sanitizer.json"),
    enable_anomaly=False,
    enable_intent=False,
    enable_sandbox=False,
)
asyncio.run(run_benchmark(config))
PY
```

For a same-split public classifier baseline on the converted samples:

```bash
uv run --with transformers --with torch python scripts/run_protectai_baseline.py \
  --attacks data/external/agentdojo/attacks \
  --benign data/external/agentdojo/benign \
  --output results/protectai_agentdojo.json
```

The ProtectAI baseline scans concatenated tool-response text only. It does not
model action-chain anomaly detection, intent verification, or sandbox policy.

## Adaptive-Evasion Stress Check

The sanitizer includes normalization for common deterministic evasion patterns,
including zero-width characters, simple homoglyph substitutions, and
separator-obfuscated security keywords. To run a supplemental sanitizer-only
stress check over generated or imported attack samples:

```bash
uv run python scripts/run_adaptive_stress.py \
  --attacks data/generated/attacks \
  --benign data/generated/benign \
  --output results/adaptive_stress.json
```

This harness mutates known attack samples with deterministic obfuscations and
reports both raw per-mutation detection rates and retention among samples that
the sanitizer detects before mutation. The retention view is useful because some
benchmark samples are intentionally handled by other ShieldAgent layers rather
than by sanitizer patterns. This is not a substitute for a full adaptive human
red-team or operational trace evaluation.

## Optional LLM-Backed Runs

LLM-backed validation runs require your own API key. Copy `.env.example` to `.env` and set the relevant provider key.

```bash
uv run python scripts/run_experiments.py benchmark --attacks data/generated/attacks --benign data/generated/benign --use-llm
```

## Notes on Exact Paper Metrics

The paper reports results from the authors' benchmark run over 1,000 generated attack samples and 500 benign samples. This public artifact includes the implementation, generator, representative samples, pretrained GNN weights, external benchmark adapters, adaptive-evasion checks, and commands needed to regenerate benchmark-style datasets. Exact metric reproduction may still require the frozen benchmark snapshot and runtime configuration associated with a specific release.
