# Released Model Artifact

This directory contains the pretrained ShieldAgent GNN anomaly-detector weights
released with the IEEE CONECCT 2026 public artifact.

- `shieldagent_gnn.pt`: GAT-based action-chain anomaly detector trained on the
  generated benchmark snapshot produced by `scripts/generate_attacks.py` with
  seed `1043`.

The model is provided for inspection and reproducibility support. For exact
comparisons, regenerate the benchmark snapshot with the same seed or train a new
model using:

```bash
uv run shieldagent generate-data --output data/generated_train --num 250 --benign 500 --seed 1043
uv run shieldagent train-gnn --data data/generated_train --output models/shieldagent_gnn.pt --epochs 30
```
