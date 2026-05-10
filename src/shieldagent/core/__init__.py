"""Core module initialization."""

from shieldagent.core.agent import ShieldAgent, create_agent, ProcessingContext
from shieldagent.core.config import ShieldAgentConfig
from shieldagent.core.types import (
    ThreatLevel,
    AttackType,
    ToolCall,
    ToolChain,
    ThreatDetails,
    DefenseResult,
    AgentResult,
    UserIntent,
    AttackSample,
    BenignSample,
)

__all__ = [
    "ShieldAgent",
    "create_agent",
    "ProcessingContext",
    "ShieldAgentConfig",
    "ThreatLevel",
    "AttackType",
    "ToolCall",
    "ToolChain",
    "ThreatDetails",
    "DefenseResult",
    "AgentResult",
    "UserIntent",
    "AttackSample",
    "BenignSample",
]
