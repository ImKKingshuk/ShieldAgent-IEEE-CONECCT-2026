"""Benchmark runner for ShieldAgent evaluation."""

import asyncio
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Iterator
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from shieldagent.core.agent import ShieldAgent
from shieldagent.core.config import ShieldAgentConfig
from shieldagent.core.types import AttackSample, BenignSample, AttackType, ToolChain, ToolCall
from shieldagent.evaluation.metrics import SecurityMetrics
from shieldagent.defense.intent import ActionProposal


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    attack_dataset_path: Path
    benign_dataset_path: Path
    output_path: Path
    
    # Ablation settings
    enable_sanitizer: bool = True
    enable_anomaly: bool = True
    enable_intent: bool = True
    enable_sandbox: bool = True
    
    # Experiment settings
    num_runs: int = 1
    seed: int = 42
    use_llm: bool = False  # Use LLM tool-calling for evaluation (requires API)
    
    @property
    def ablation_name(self) -> str:
        """Generate name based on enabled components."""
        components = []
        if self.enable_sanitizer:
            components.append("S")
        if self.enable_anomaly:
            components.append("A")
        if self.enable_intent:
            components.append("I")
        if self.enable_sandbox:
            components.append("X")
        return "+".join(components) if components else "None"


def load_attack_samples(path: Path) -> Iterator[AttackSample]:
    """Load attack samples from JSON files."""
    if not path.exists():
        logger.warning(f"Attack dataset path not found: {path}")
        return
    
    for file in path.glob("*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
            
            yield AttackSample(
                id=data.get("id", file.stem),
                attack_type=AttackType(data.get("attack_type", "unknown")),
                prompt=data["prompt"],
                tool_responses=data.get("tool_responses", []),
                tool_sequence=data.get("tool_sequence", []),
                expected_detected=data.get("expected_detected", True),
                description=data.get("description", ""),
            )
        except Exception as e:
            logger.error(f"Failed to load attack sample {file}: {e}")


def load_benign_samples(path: Path) -> Iterator[BenignSample]:
    """Load benign samples from JSON files."""
    if not path.exists():
        logger.warning(f"Benign dataset path not found: {path}")
        return
    
    for file in path.glob("*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
            
            yield BenignSample(
                id=data.get("id", file.stem),
                prompt=data["prompt"],
                tool_responses=data.get("tool_responses", []),
                expected_safe=data.get("expected_safe", True),
                description=data.get("description", ""),
            )
        except Exception as e:
            logger.error(f"Failed to load benign sample {file}: {e}")


async def run_benchmark(config: BenchmarkConfig) -> dict:
    """
    Run a complete benchmark evaluation.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"Starting benchmark: {config.ablation_name}")
    
    # Reproducibility
    random.seed(config.seed)
    try:
        import numpy as np
        np.random.seed(config.seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(config.seed)
    except Exception:
        pass

    # Create agent with specified configuration
    agent_config = ShieldAgentConfig.from_env()
    agent_config.sanitizer.enabled = config.enable_sanitizer
    agent_config.anomaly_detector.enabled = config.enable_anomaly
    agent_config.intent_verifier.enabled = config.enable_intent
    agent_config.sandbox.enabled = config.enable_sandbox
    
    agent = ShieldAgent(agent_config)
    metrics = SecurityMetrics()
    
    # Load samples
    attack_samples = list(load_attack_samples(config.attack_dataset_path))
    benign_samples = list(load_benign_samples(config.benign_dataset_path))
    
    logger.info(f"Loaded {len(attack_samples)} attack and {len(benign_samples)} benign samples")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        
        # Test attack samples
        attack_task = progress.add_task("Testing attacks...", total=len(attack_samples))
        
        for sample in attack_samples:
            detected = False
            threat_details = None
            latency_ms = 0.0
            
            if config.use_llm:
                # LLM-driven evaluation (non-deterministic, requires API)
                result = await agent.process(user_prompt=sample.prompt)
                detected = result.attack_detected
                threat_details = result.threat_details
                latency_ms = result.total_latency_ms
            else:
                # Dataset-driven evaluation (deterministic)
                total_processing_ms = 0.0
                # Dataset-driven evaluation (deterministic)
                if agent_config.intent_verifier.enabled:
                    intent = agent.intent_verifier.extract_user_intent(sample.prompt)
                    actions = []
                    for tr in sample.tool_responses:
                        actions.append(ActionProposal(
                            action_type="tool_call",
                            description=f"Execute {tr.get('tool_name', 'unknown')}",
                            tool_name=tr.get("tool_name", "unknown"),
                            arguments={},
                        ))
                    if actions:
                        intent_result = agent.intent_verifier.verify_action_chain(intent, actions)
                        total_processing_ms += intent_result.processing_time_ms
                        if not intent_result.is_safe:
                            detected = True
                            threat_details = intent_result.threat_detected
                
                if agent_config.anomaly_detector.enabled:
                    tool_sequence = sample.tool_sequence or [tr.get("tool_name", "unknown") for tr in sample.tool_responses]
                    chain = ToolChain()
                    for tool_name in tool_sequence:
                        chain.add_call(ToolCall(tool_name=tool_name, arguments={}))
                    if len(chain) > 1:
                        anomaly_result = agent.anomaly_detector.detect(chain)
                        total_processing_ms += anomaly_result.processing_time_ms
                        if not anomaly_result.is_safe:
                            detected = True
                            threat_details = threat_details or anomaly_result.threat_detected
                
                if agent_config.sanitizer.enabled:
                    for tool_resp in sample.tool_responses:
                        sanitizer_result = await agent.process_tool_response(
                            tool_name=tool_resp.get("tool_name", "unknown"),
                            tool_response=tool_resp.get("response", ""),
                        )
                        total_processing_ms += sanitizer_result.processing_time_ms
                        if not sanitizer_result.is_safe:
                            detected = True
                            threat_details = threat_details or sanitizer_result.threat_detected
                            break

                if agent_config.sandbox.enabled:
                    for tool_resp in sample.tool_responses:
                        tool_name = tool_resp.get("tool_name", "unknown")
                        if agent.sandbox.is_high_risk(tool_name):
                            tool_call = ToolCall(tool_name=tool_name, arguments={})

                            def _stub_executor(_: dict) -> str:
                                return tool_resp.get("response", "")

                            sandbox_result = await agent.sandbox.execute_with_sandbox(
                                tool_call=tool_call,
                                executor_func=_stub_executor,
                            )
                            total_processing_ms += sandbox_result.processing_time_ms
                            if not sandbox_result.is_safe:
                                detected = True
                                threat_details = threat_details or sandbox_result.threat_detected
                                break
                
                latency_ms = total_processing_ms
            
            if detected and threat_details:
                confidence = threat_details.confidence
            else:
                confidence = 1.0 if detected else 0.0
            metrics.record_attack(
                detected=detected,
                attack_type=sample.attack_type,
                confidence=confidence,
                latency_ms=latency_ms,
            )
            
            progress.advance(attack_task)
        
        # Test benign samples
        benign_task = progress.add_task("Testing benign...", total=len(benign_samples))
        
        for sample in benign_samples:
            flagged = False
            latency_ms = 0.0
            
            if config.use_llm:
                result = await agent.process(user_prompt=sample.prompt)
                flagged = result.attack_detected
                latency_ms = result.total_latency_ms
            else:
                total_processing_ms = 0.0
                if agent_config.intent_verifier.enabled:
                    intent = agent.intent_verifier.extract_user_intent(sample.prompt)
                    actions = []
                    for tr in sample.tool_responses:
                        actions.append(ActionProposal(
                            action_type="tool_call",
                            description=f"Execute {tr.get('tool_name', 'unknown')}",
                            tool_name=tr.get("tool_name", "unknown"),
                            arguments={},
                        ))
                    if actions:
                        intent_result = agent.intent_verifier.verify_action_chain(intent, actions)
                        total_processing_ms += intent_result.processing_time_ms
                        if not intent_result.is_safe:
                            flagged = True
                
                if agent_config.anomaly_detector.enabled:
                    chain = ToolChain()
                    tool_sequence = [tr.get("tool_name", "unknown") for tr in sample.tool_responses]
                    for tool_name in tool_sequence:
                        chain.add_call(ToolCall(tool_name=tool_name, arguments={}))
                    if len(chain) > 1:
                        anomaly_result = agent.anomaly_detector.detect(chain)
                        total_processing_ms += anomaly_result.processing_time_ms
                        if not anomaly_result.is_safe:
                            flagged = True
                
                if agent_config.sanitizer.enabled:
                    for tool_resp in sample.tool_responses:
                        sanitizer_result = await agent.process_tool_response(
                            tool_name=tool_resp.get("tool_name", "unknown"),
                            tool_response=tool_resp.get("response", ""),
                        )
                        total_processing_ms += sanitizer_result.processing_time_ms
                        if not sanitizer_result.is_safe:
                            flagged = True
                            break

                if agent_config.sandbox.enabled:
                    for tool_resp in sample.tool_responses:
                        tool_name = tool_resp.get("tool_name", "unknown")
                        if agent.sandbox.is_high_risk(tool_name):
                            tool_call = ToolCall(tool_name=tool_name, arguments={})

                            def _stub_executor(_: dict) -> str:
                                return tool_resp.get("response", "")

                            sandbox_result = await agent.sandbox.execute_with_sandbox(
                                tool_call=tool_call,
                                executor_func=_stub_executor,
                            )
                            total_processing_ms += sandbox_result.processing_time_ms
                            if not sandbox_result.is_safe:
                                flagged = True
                                break
                
                latency_ms = total_processing_ms
            
            metrics.record_benign(
                flagged=flagged,
                latency_ms=latency_ms,
            )
            
            progress.advance(benign_task)
    
    # Compute results
    results = metrics.compute()
    results["config"] = {
        "ablation_name": config.ablation_name,
        "enable_sanitizer": config.enable_sanitizer,
        "enable_anomaly": config.enable_anomaly,
        "enable_intent": config.enable_intent,
        "enable_sandbox": config.enable_sandbox,
        "num_attack_samples": len(attack_samples),
        "num_benign_samples": len(benign_samples),
    }
    
    # Save results
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {config.output_path}")
    print(metrics.summary())
    
    agent.shutdown()
    
    return results


async def run_ablation_study(
    attack_path: Path,
    benign_path: Path,
    output_dir: Path,
    use_llm: bool = False,
) -> dict[str, dict]:
    """
    Run complete ablation study with all component combinations.
    
    Tests:
    - Full ShieldAgent (all components)
    - Each component individually
    - Various component combinations
    """
    ablation_configs = [
        # Full model
        (True, True, True, True, "full"),
        
        # Individual components
        (True, False, False, False, "sanitizer_only"),
        (False, True, False, False, "anomaly_only"),
        (False, False, True, False, "intent_only"),
        
        # Component pairs
        (True, True, False, False, "san_anom"),
        (True, False, True, False, "san_intent"),
        (False, True, True, False, "anom_intent"),
        
        # Without specific components
        (False, True, True, True, "no_sanitizer"),
        (True, False, True, True, "no_anomaly"),
        (True, True, False, True, "no_intent"),
        
        # Baseline (no defense)
        (False, False, False, False, "baseline"),
    ]
    
    all_results = {}
    
    for san, anom, intent, sandbox, name in ablation_configs:
        config = BenchmarkConfig(
            attack_dataset_path=attack_path,
            benign_dataset_path=benign_path,
            output_path=output_dir / f"{name}.json",
            enable_sanitizer=san,
            enable_anomaly=anom,
            enable_intent=intent,
            enable_sandbox=sandbox,
            use_llm=use_llm,
        )
        
        results = await run_benchmark(config)
        all_results[name] = results
    
    # Save combined results
    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results
