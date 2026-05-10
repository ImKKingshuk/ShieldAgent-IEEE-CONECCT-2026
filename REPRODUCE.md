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

For ablations:

```bash
uv run python scripts/run_experiments.py benchmark --attacks data/generated/attacks --benign data/generated/benign --ablation
```

## Optional LLM-Backed Runs

LLM-backed validation runs require your own API key. Copy `.env.example` to `.env` and set the relevant provider key.

```bash
uv run python scripts/run_experiments.py benchmark --attacks data/generated/attacks --benign data/generated/benign --use-llm
```

## Notes on Exact Paper Metrics

The paper reports results from the authors' benchmark run over 1,000 generated attack samples and 500 benign samples. This public artifact includes the implementation, generator, representative samples, and commands needed to regenerate benchmark-style datasets. Exact metric reproduction may require the frozen benchmark snapshot, runtime configuration, and pretrained anomaly-detection artifacts associated with a specific release.
