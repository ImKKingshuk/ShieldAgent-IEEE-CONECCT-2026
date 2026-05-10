"""Main experiment runner for ShieldAgent evaluation."""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from loguru import logger

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shieldagent.evaluation.benchmark import (
    BenchmarkConfig,
    run_benchmark,
    run_ablation_study,
)

app = typer.Typer(help="ShieldAgent Experiment Runner")
console = Console()


@app.command()
def main(
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
    config_file: Path = typer.Option(
        None,
        "--config", "-c",
        help="Path to experiment config YAML"
    ),
    use_llm: bool = typer.Option(
        False,
        "--use-llm",
        help="Use LLM tool-calling during evaluation (requires API)"
    ),
):
    """Run ShieldAgent benchmark experiments."""
    
    console.print("\n[bold blue]ShieldAgent Experiment Runner[/bold blue]")
    console.print("=" * 50)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"Attack dataset: {attack_path}")
    console.print(f"Benign dataset: {benign_path}")
    console.print(f"Output: {output_path}")
    console.print()
    
    if ablation:
        # Run full ablation study
        console.print("[bold]Running ablation study...[/bold]\n")
        results = asyncio.run(run_ablation_study(
            attack_path=attack_path,
            benign_path=benign_path,
            output_dir=output_path,
            use_llm=use_llm,
        ))
        
        # Display comparison table
        display_ablation_results(results)
    else:
        # Run single benchmark with full model
        console.print("[bold]Running full model benchmark...[/bold]\n")
        config = BenchmarkConfig(
            attack_dataset_path=attack_path,
            benign_dataset_path=benign_path,
            output_path=output_path / "full_model.json",
            use_llm=use_llm,
        )
        results = asyncio.run(run_benchmark(config))
    
    console.print(f"\n[green]Results saved to {output_path}[/green]")


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
    config_file: Path = typer.Option(
        None,
        "--config", "-c",
        help="Path to experiment config YAML"
    ),
    use_llm: bool = typer.Option(
        False,
        "--use-llm",
        help="Use LLM tool-calling during evaluation (requires API)"
    ),
):
    """Alias for main() to match documentation."""
    return main(
        attack_path=attack_path,
        benign_path=benign_path,
        output_dir=output_dir,
        ablation=ablation,
        config_file=config_file,
        use_llm=use_llm,
    )


def display_ablation_results(results: dict):
    """Display ablation study results in a table."""
    
    table = Table(title="Ablation Study Results")
    
    table.add_column("Configuration", style="cyan")
    table.add_column("Detection Rate", justify="right")
    table.add_column("False Positive Rate", justify="right")
    table.add_column("F1 Score", justify="right")
    table.add_column("Latency (ms)", justify="right")
    
    for name, data in results.items():
        overall = data.get("overall", {})
        benign = data.get("benign", {})
        latency = data.get("latency", {})
        
        detection_rate = overall.get("recall", 0) * 100
        fp_rate = benign.get("false_positive_rate", 0) * 100
        f1 = overall.get("f1", 0)
        mean_latency = latency.get("mean_ms", 0)
        
        # Color code based on performance
        dr_str = f"[green]{detection_rate:.1f}%[/green]" if detection_rate > 90 else f"[yellow]{detection_rate:.1f}%[/yellow]" if detection_rate > 70 else f"[red]{detection_rate:.1f}%[/red]"
        fp_str = f"[green]{fp_rate:.1f}%[/green]" if fp_rate < 5 else f"[yellow]{fp_rate:.1f}%[/yellow]" if fp_rate < 10 else f"[red]{fp_rate:.1f}%[/red]"
        
        table.add_row(
            name,
            dr_str,
            fp_str,
            f"{f1:.3f}",
            f"{mean_latency:.1f}",
        )
    
    console.print(table)


@app.command()
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
):
    """Generate attack and benign datasets."""
    from generate_attacks import generate_dataset
    
    console.print(f"\n[bold]Generating dataset with {num_samples} samples per category...[/bold]\n")
    stats = generate_dataset(output_dir, num_samples)
    
    console.print("\n[bold green]Dataset generation complete![/bold green]")


@app.command()
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
    from shieldagent.defense.anomaly import ActionChainAnomalyDetector
    from shieldagent.core.config import AnomalyDetectorConfig
    from shieldagent.core.types import ToolChain, ToolCall
    from shieldagent.evaluation.benchmark import load_attack_samples, load_benign_samples
    from datetime import datetime
    
    console.print("\n[bold]Training GNN Anomaly Detector[/bold]\n")
    
    # Load data
    attack_samples = list(load_attack_samples(data_dir / "attacks"))
    benign_samples = list(load_benign_samples(data_dir / "benign"))
    
    console.print(f"Loaded {len(attack_samples)} attack samples")
    console.print(f"Loaded {len(benign_samples)} benign samples")
    
    # Convert to tool chains
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
    
    console.print(f"\nTraining on {len(attack_chains)} attack chains and {len(benign_chains)} benign chains")
    
    # Train
    detector = ActionChainAnomalyDetector(AnomalyDetectorConfig(enabled=True))
    metrics = detector.train_model(benign_chains, attack_chains, epochs=epochs)
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save_model(output_path)
    
    console.print(f"\n[green]Model saved to {output_path}[/green]")
    console.print(f"Final accuracy: {metrics['accuracies'][-1]:.4f}")


if __name__ == "__main__":
    app()
