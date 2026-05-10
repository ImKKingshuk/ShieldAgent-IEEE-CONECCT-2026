# ShieldAgent Data

This directory contains representative sample records and instructions for generating larger benchmark datasets.

## Included Samples

`data/samples/` contains a compact, inspectable sample set:

- `attacks/`: representative malicious tool-use scenarios.
- `benign/`: representative safe tool-use scenarios.

These samples are intended for pipeline validation and documentation. They are not the full paper benchmark.

## Generating Data

```bash
uv run python scripts/generate_attacks.py --output data/generated --samples 2 --benign 4 --seed 1043
```

For a larger paper-scale generated dataset:

```bash
uv run python scripts/generate_attacks.py --output data/generated_full --samples 250 --benign 500 --seed 1043
```

Generated folders are ignored by default so local experiments do not affect the repository state.
