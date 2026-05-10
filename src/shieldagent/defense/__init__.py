"""Defense module initialization."""

from shieldagent.defense.sanitizer import (
    ToolResponseSanitizer,
    SanitizationPattern,
    DEFAULT_INJECTION_PATTERNS,
    DEFAULT_COMMAND_PATTERNS,
    DEFAULT_CONFIG_PATTERNS,
)
from shieldagent.defense.intent import (
    IntentVerifier,
    ActionProposal,
)
from shieldagent.defense.anomaly import (
    ActionChainAnomalyDetector,
    ActionChainGNN,
    ToolCallGraphEncoder,
    KNOWN_ATTACK_PATTERNS,
)
from shieldagent.defense.sandbox import (
    SandboxedExecutor,
    SandboxState,
    ExecutionResult,
    HIGH_RISK_TOOLS,
    BLOCKED_TOOLS,
)

__all__ = [
    # Sanitizer
    "ToolResponseSanitizer",
    "SanitizationPattern",
    "DEFAULT_INJECTION_PATTERNS",
    "DEFAULT_COMMAND_PATTERNS",
    "DEFAULT_CONFIG_PATTERNS",
    
    # Intent
    "IntentVerifier",
    "ActionProposal",
    
    # Anomaly
    "ActionChainAnomalyDetector",
    "ActionChainGNN",
    "ToolCallGraphEncoder",
    "KNOWN_ATTACK_PATTERNS",
    
    # Sandbox
    "SandboxedExecutor",
    "SandboxState",
    "ExecutionResult",
    "HIGH_RISK_TOOLS",
    "BLOCKED_TOOLS",
]
