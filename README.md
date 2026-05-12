# ShieldAgent: Defense-in-Depth Security Framework for Agentic AI Tool Use

ShieldAgent is a research prototype for studying defense-in-depth mechanisms for tool-using agentic AI systems. It targets indirect prompt injection, malicious tool-call sequences, memory poisoning, and configuration manipulation in agent workflows.

This repository is the public artifact for the IEEE CONECCT 2026 paper:

> ShieldAgent: Defense-in-Depth Security Framework for Agentic AI Tool Use

## Artifact Contents

- Source code for the four defense layers:
  - tool response sanitization
  - intent verification
  - action-chain anomaly detection
  - sandboxed execution
- MCP client/server integration wrappers.
- Dataset-generation scripts for representative attack and benign scenarios.
- Import adapters for supplemental public-benchmark stress checks using AgentDojo
  and ShieldLM.
- Released GNN anomaly-detector weights in `models/shieldagent_gnn.pt`.
- Unit tests for the core defense modules.
- Reproducibility instructions using `uv`.
- A small representative sample dataset in `data/samples/`.
- Docker-based environment specification for reproducible local execution.

## Scope and Limitations

The repository does not include:

- API keys or private environment files.
- Generated experiment results.
- Conference submission files, review exports, or camera-ready administration documents.
- Frozen benchmark snapshots.

The repository is intended to support inspection, local testing, benchmark-style regeneration, and supplemental checks on public prompt-injection corpora. Exact reproduction of paper-level aggregate metrics may require the frozen benchmark snapshot associated with a specific release.

## Installation

```bash
git clone https://github.com/ImKKingshuk/ShieldAgent-IEEE-CONECCT-2026.git
cd ShieldAgent-IEEE-CONECCT-2026
uv sync
```

For LLM-backed runs, copy the example environment file and add your own provider key:

```bash
cp .env.example .env
```

The deterministic tests and benchmark paths do not require an API key.

## Validation Commands

```bash
uv run pytest tests/ -v
uv run python scripts/generate_attacks.py --output data/generated --samples 2 --benign 4 --seed 1043
uv run python scripts/run_experiments.py benchmark --attacks data/generated/attacks --benign data/generated/benign
uv run --with agentdojo python scripts/import_agentdojo.py --output data/external/agentdojo
uv run --with datasets python scripts/import_shieldlm.py --output data/external/shieldlm
```

## Project Structure

```text
shieldagent/
├── src/shieldagent/        # framework implementation
├── configs/                # defense configuration and pattern library
├── scripts/                # dataset generation and experiment runners
├── tests/                  # unit tests
├── data/samples/           # representative sample records
├── README.md
├── REPRODUCE.md
├── ARTIFACT.md
├── pyproject.toml
└── uv.lock
```

## Citation

```bibtex
@inproceedings{shieldagent2026,
  title={ShieldAgent: Defense-in-Depth Security Framework for Agentic AI Tool Use},
  author={Kingshuk Mondal},
  booktitle={IEEE CONECCT},
  year={2026}
}
```

## License

This artifact is released under the MIT License. See `LICENSE`.
