"""
ShieldAgent: Defense-in-Depth Security Framework for Agentic AI Tool Use

A defense-in-depth research prototype for studying tool-use attacks,
prompt injection, and action-chain exploitation in agentic AI systems.
"""

from shieldagent.core.agent import ShieldAgent
from shieldagent.core.config import ShieldAgentConfig
from shieldagent.core.types import (
    AgentResult,
    ThreatLevel,
    AttackType,
    DefenseResult,
)

__version__ = "0.1.0"
__author__ = "Kingshuk Mondal"

__all__ = [
    "ShieldAgent",
    "ShieldAgentConfig",
    "AgentResult",
    "ThreatLevel",
    "AttackType",
    "DefenseResult",
]
