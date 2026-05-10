"""Core type definitions for ShieldAgent."""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


class ThreatLevel(Enum):
    """Threat severity levels."""
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class AttackType(Enum):
    """Categories of attacks detected by ShieldAgent."""
    NONE = "none"
    PROMPT_INJECTION = "prompt_injection"
    TOOL_CHAIN_EXPLOIT = "tool_chain_exploit"
    MEMORY_POISONING = "memory_poisoning"
    CONFIG_MANIPULATION = "config_manipulation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    UNKNOWN = "unknown"


@dataclass
class ToolCall:
    """Represents a single tool invocation."""
    tool_name: str
    arguments: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    response: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class ToolChain:
    """Represents a sequence of tool calls."""
    calls: list[ToolCall] = field(default_factory=list)
    
    def add_call(self, call: ToolCall) -> None:
        self.calls.append(call)
    
    def get_sequence(self) -> list[str]:
        """Get tool names in sequence."""
        return [call.tool_name for call in self.calls]
    
    def __len__(self) -> int:
        return len(self.calls)


@dataclass
class ThreatDetails:
    """Details about a detected threat."""
    attack_type: AttackType
    threat_level: ThreatLevel
    description: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    affected_tools: list[str] = field(default_factory=list)
    mitigation_applied: Optional[str] = None


@dataclass
class DefenseResult:
    """Result from a single defense module."""
    module_name: str
    is_safe: bool
    threat_detected: Optional[ThreatDetails] = None
    processing_time_ms: float = 0.0
    sanitized_content: Optional[str] = None


@dataclass
class AgentResult:
    """Final result from ShieldAgent processing."""
    success: bool
    output: Optional[str] = None
    attack_detected: bool = False
    threat_details: Optional[ThreatDetails] = None
    defense_results: list[DefenseResult] = field(default_factory=list)
    tool_chain: Optional[ToolChain] = None
    total_latency_ms: float = 0.0
    blocked: bool = False
    error: Optional[str] = None


@dataclass
class UserIntent:
    """Represents the user's original intent."""
    prompt: str
    embedding: Optional[list[float]] = None
    extracted_goals: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)


@dataclass
class AttackSample:
    """A single attack sample for evaluation."""
    id: str
    attack_type: AttackType
    prompt: str
    tool_responses: list[dict[str, Any]]
    tool_sequence: list[str] = field(default_factory=list)
    expected_detected: bool = True
    description: str = ""
    severity: ThreatLevel = ThreatLevel.HIGH


@dataclass
class BenignSample:
    """A benign interaction sample for evaluation."""
    id: str
    prompt: str
    tool_responses: list[dict[str, Any]]
    expected_safe: bool = True
    description: str = ""
