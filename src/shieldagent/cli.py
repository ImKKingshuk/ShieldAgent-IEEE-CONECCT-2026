"""Command-line interface for ShieldAgent."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from shieldagent.evaluation.benchmark import (
    BenchmarkConfig,
    run_benchmark,
    run_ablation_study,
    load_attack_samples,
    load_benign_samples,
)
from shieldagent.defense.anomaly import ActionChainAnomalyDetector
from shieldagent.core.config import AnomalyDetectorConfig
from shieldagent.core.types import ToolChain, ToolCall

app = typer.Typer(help="ShieldAgent CLI")
console = Console()


@app.command("benchmark")
def benchmark(
    attack_path: Path = typer.Option(
        Path("data/attacks"),
        "--attacks", "-a",
        help="Path to attack dataset"
    ),
    benign_path: Path = typer.Option(
        Path("data/benign"),
        "--benign", "-b",
        help="Path to benign dataset"
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output", "-o",
        help="Output directory for results"
    ),
    ablation: bool = typer.Option(
        False,
        "--ablation",
        help="Run full ablation study"
    ),
    use_llm: bool = typer.Option(
        False,
        "--use-llm",
        help="Use LLM tool-calling during evaluation (requires API)"
    ),
):
    """Run ShieldAgent benchmark experiments."""
    console.print("\n[bold blue]ShieldAgent Benchmark[/bold blue]")
    console.print("=" * 50)

    output_path = output_dir / _timestamp()
    output_path.mkdir(parents=True, exist_ok=True)

    if ablation:
        console.print("[bold]Running ablation study...[/bold]\n")
        results = asyncio.run(run_ablation_study(
            attack_path=attack_path,
            benign_path=benign_path,
            output_dir=output_path,
            use_llm=use_llm,
        ))
        console.print(f"\n[green]Ablation results saved to {output_path}[/green]")
    else:
        console.print("[bold]Running full model benchmark...[/bold]\n")
        config = BenchmarkConfig(
            attack_dataset_path=attack_path,
            benign_dataset_path=benign_path,
            output_path=output_path / "full_model.json",
            use_llm=use_llm,
        )
        asyncio.run(run_benchmark(config))
        console.print(f"\n[green]Results saved to {output_path}[/green]")


@app.command("generate-data")
def generate_data(
    output_dir: Path = typer.Option(
        Path("data"),
        "--output", "-o",
        help="Output directory for generated data"
    ),
    num_samples: int = typer.Option(
        50,
        "--num", "-n",
        help="Number of samples per attack category"
    ),
    benign_samples: Optional[int] = typer.Option(
        None,
        "--benign", "-b",
        help="Number of benign samples (optional)"
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed", "-s",
        help="Random seed for reproducibility"
    ),
):
    """Generate attack and benign datasets."""
    generator = _load_generator()
    if not generator:
        console.print("[red]Could not load dataset generator. Ensure scripts/generate_attacks.py exists.[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]Generating dataset with {num_samples} samples per category...[/bold]\n")
    generator.generate_dataset(output_dir, num_samples, benign_samples, seed)
    console.print("\n[bold green]Dataset generation complete![/bold green]")


@app.command("train-gnn")
def train_gnn(
    data_dir: Path = typer.Option(
        Path("data"),
        "--data", "-d",
        help="Directory containing attack and benign datasets"
    ),
    output_path: Path = typer.Option(
        Path("models/gnn_model.pt"),
        "--output", "-o",
        help="Path to save trained model"
    ),
    epochs: int = typer.Option(
        100,
        "--epochs", "-e",
        help="Number of training epochs"
    ),
):
    """Train the GNN anomaly detector."""
    console.print("\n[bold]Training GNN Anomaly Detector[/bold]\n")

    attack_samples = list(load_attack_samples(data_dir / "attacks"))
    benign_samples = list(load_benign_samples(data_dir / "benign"))

    console.print(f"Loaded {len(attack_samples)} attack samples")
    console.print(f"Loaded {len(benign_samples)} benign samples")

    attack_chains = []
    for sample in attack_samples:
        chain = ToolChain()
        for resp in sample.tool_responses:
            chain.add_call(ToolCall(
                tool_name=resp.get("tool_name", "unknown"),
                arguments={},
            ))
        if len(chain) > 1:
            attack_chains.append(chain)

    benign_chains = []
    for sample in benign_samples:
        chain = ToolChain()
        for resp in sample.tool_responses:
            chain.add_call(ToolCall(
                tool_name=resp.get("tool_name", "unknown"),
                arguments={},
            ))
        benign_chains.append(chain)

    detector = ActionChainAnomalyDetector(AnomalyDetectorConfig(enabled=True))
    metrics = detector.train_model(benign_chains, attack_chains, epochs=epochs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save_model(output_path)

    console.print(f"\n[green]Model saved to {output_path}[/green]")
    console.print(f"Final accuracy: {metrics['accuracies'][-1]:.4f}")


def _load_generator():
    """Load the dataset generator from scripts/generate_attacks.py."""
    script_path = Path("scripts/generate_attacks.py")
    if not script_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("generate_attacks", script_path)
    if not spec or not spec.loader:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    app()
