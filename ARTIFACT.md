# Artifact Description

## Purpose

This repository provides the public artifact for the IEEE CONECCT 2026 paper "ShieldAgent: Defense-in-Depth Security Framework for Agentic AI Tool Use."

The artifact is designed to support independent inspection of the implementation, execution of the unit-test suite, generation of representative benchmark-style data, and local reproduction of the evaluation workflow.

## Components

- `src/shieldagent/`: implementation of the ShieldAgent orchestration layer and defense modules.
- `configs/attack_patterns.yaml`: pattern definitions used by the tool-response sanitizer.
- `scripts/generate_attacks.py`: generator for synthetic attack and benign tool-use scenarios.
- `scripts/import_agentdojo.py`: imports AgentDojo task and injection goals into a compact ShieldAgent stress-test set.
- `scripts/import_shieldlm.py`: imports a balanced ShieldLM prompt-injection subset for supplemental sanitizer evaluation.
- `scripts/run_experiments.py`: benchmark and ablation runner.
- `models/shieldagent_gnn.pt`: released pretrained anomaly-detector weights.
- `tests/`: unit tests for the core components.
- `data/samples/`: compact representative samples for inspection and pipeline validation.
- `REPRODUCE.md`: setup and execution instructions.
- `Dockerfile`: container specification for a reproducible execution environment.

## Reproducibility Scope

The repository supports deterministic unit testing, local generation of benchmark-style datasets, reuse of released GNN weights, and supplemental public-benchmark stress checks. Exact reproduction of paper-level aggregate metrics may require a frozen benchmark snapshot and the same runtime configuration associated with a specific release.

No private submission documents, review files, real credentials, API keys, or personal data are included in this artifact.

## Ethical Use

The attack examples are synthetic and intended for defensive research, evaluation, and education. They use reserved domains such as `attacker.invalid` and placeholder credentials such as `EXAMPLE_API_KEY_REDACTED`.
