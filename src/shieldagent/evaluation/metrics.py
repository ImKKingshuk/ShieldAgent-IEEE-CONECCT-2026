"""Evaluation metrics for ShieldAgent."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

from shieldagent.core.types import AttackType, ThreatLevel


@dataclass
class AttackMetrics:
    """Metrics for a single attack type."""
    attack_type: AttackType
    total: int = 0
    detected: int = 0
    missed: int = 0
    
    @property
    def detection_rate(self) -> float:
        return self.detected / max(1, self.total)


@dataclass
class BenignMetrics:
    """Metrics for benign samples."""
    total: int = 0
    correctly_passed: int = 0
    false_positives: int = 0
    
    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / max(1, self.total)


@dataclass
class LatencyMetrics:
    """Latency measurement metrics."""
    samples: list[float] = field(default_factory=list)
    
    def add_sample(self, latency_ms: float) -> None:
        self.samples.append(latency_ms)
    
    @property
    def mean(self) -> float:
        return np.mean(self.samples) if self.samples else 0.0
    
    @property
    def std(self) -> float:
        return np.std(self.samples) if self.samples else 0.0
    
    @property
    def p50(self) -> float:
        return np.percentile(self.samples, 50) if self.samples else 0.0
    
    @property
    def p95(self) -> float:
        return np.percentile(self.samples, 95) if self.samples else 0.0
    
    @property
    def p99(self) -> float:
        return np.percentile(self.samples, 99) if self.samples else 0.0


class SecurityMetrics:
    """
    Comprehensive security evaluation metrics for ShieldAgent.
    
    Tracks:
    - Per-attack-type detection rates
    - False positive rates on benign samples
    - Overall precision, recall, F1
    - Latency overhead
    """
    
    def __init__(self):
        self.attack_metrics: dict[AttackType, AttackMetrics] = {
            at: AttackMetrics(attack_type=at) for at in AttackType
        }
        self.benign_metrics = BenignMetrics()
        self.latency = LatencyMetrics()
        
        # Raw predictions for sklearn metrics
        self._y_true: list[int] = []
        self._y_pred: list[int] = []
        self._y_scores: list[float] = []
    
    def record_attack(
        self,
        detected: bool,
        attack_type: AttackType,
        confidence: float = 1.0,
        latency_ms: float = 0.0,
    ) -> None:
        """Record result for an attack sample."""
        metrics = self.attack_metrics[attack_type]
        metrics.total += 1
        
        if detected:
            metrics.detected += 1
        else:
            metrics.missed += 1
        
        self._y_true.append(1)  # Attack = positive class
        self._y_pred.append(1 if detected else 0)
        self._y_scores.append(confidence)
        
        if latency_ms > 0:
            self.latency.add_sample(latency_ms)
    
    def record_benign(
        self,
        flagged: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """Record result for a benign sample."""
        self.benign_metrics.total += 1
        
        if flagged:
            self.benign_metrics.false_positives += 1
        else:
            self.benign_metrics.correctly_passed += 1
        
        self._y_true.append(0)  # Benign = negative class
        self._y_pred.append(1 if flagged else 0)
        self._y_scores.append(1.0 if flagged else 0.0)
        
        if latency_ms > 0:
            self.latency.add_sample(latency_ms)
    
    def compute(self) -> dict:
        """Compute all metrics and return as dictionary."""
        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)
        y_scores = np.array(self._y_scores)
        
        results = {
            "overall": {},
            "per_attack_type": {},
            "benign": {},
            "latency": {},
        }
        
        # Overall metrics
        if len(y_true) > 0:
            results["overall"] = {
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "total_samples": len(y_true),
            }
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            results["overall"]["true_positives"] = int(tp)
            results["overall"]["false_positives"] = int(fp)
            results["overall"]["true_negatives"] = int(tn)
            results["overall"]["false_negatives"] = int(fn)
            
            # ROC-AUC if we have both classes
            if len(np.unique(y_true)) > 1:
                results["overall"]["roc_auc"] = roc_auc_score(y_true, y_scores)
                
                # PR-AUC
                precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
                results["overall"]["pr_auc"] = auc(recall_arr, precision_arr)
        
        # Per attack type metrics
        for attack_type, metrics in self.attack_metrics.items():
            if metrics.total > 0:
                results["per_attack_type"][attack_type.value] = {
                    "total": metrics.total,
                    "detected": metrics.detected,
                    "missed": metrics.missed,
                    "detection_rate": metrics.detection_rate,
                }
        
        # Benign metrics
        results["benign"] = {
            "total": self.benign_metrics.total,
            "correctly_passed": self.benign_metrics.correctly_passed,
            "false_positives": self.benign_metrics.false_positives,
            "false_positive_rate": self.benign_metrics.false_positive_rate,
        }
        
        # Latency metrics
        if self.latency.samples:
            results["latency"] = {
                "mean_ms": self.latency.mean,
                "std_ms": self.latency.std,
                "p50_ms": self.latency.p50,
                "p95_ms": self.latency.p95,
                "p99_ms": self.latency.p99,
                "num_samples": len(self.latency.samples),
            }
        
        return results
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.attack_metrics = {at: AttackMetrics(attack_type=at) for at in AttackType}
        self.benign_metrics = BenignMetrics()
        self.latency = LatencyMetrics()
        self._y_true.clear()
        self._y_pred.clear()
        self._y_scores.clear()
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        results = self.compute()
        
        lines = [
            "=" * 60,
            "ShieldAgent Security Evaluation Results",
            "=" * 60,
            "",
            "Overall Metrics:",
            f"  Precision: {results['overall'].get('precision', 0):.3f}",
            f"  Recall:    {results['overall'].get('recall', 0):.3f}",
            f"  F1 Score:  {results['overall'].get('f1', 0):.3f}",
            f"  Accuracy:  {results['overall'].get('accuracy', 0):.3f}",
            "",
            "Detection by Attack Type:",
        ]
        
        for attack_type, metrics in results["per_attack_type"].items():
            lines.append(
                f"  {attack_type}: {metrics['detected']}/{metrics['total']} "
                f"({metrics['detection_rate']:.1%})"
            )
        
        lines.extend([
            "",
            "Benign Samples:",
            f"  False Positive Rate: {results['benign']['false_positive_rate']:.1%}",
            f"  ({results['benign']['false_positives']}/{results['benign']['total']})",
        ])
        
        if results.get("latency"):
            lines.extend([
                "",
                "Latency:",
                f"  Mean: {results['latency']['mean_ms']:.1f}ms",
                f"  P95:  {results['latency']['p95_ms']:.1f}ms",
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
