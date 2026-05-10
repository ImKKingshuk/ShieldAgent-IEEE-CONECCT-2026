"""MCP type definitions for ShieldAgent."""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class MCPMessageType(Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPToolRequest:
    """Request to execute a tool via MCP."""
    tool_name: str
    arguments: dict[str, Any]
    request_id: str
    session_id: Optional[str] = None


@dataclass
class MCPToolResponse:
    """Response from tool execution via MCP."""
    request_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    is_safe: bool = True
    threat_info: Optional[dict[str, Any]] = None
    processing_time_ms: float = 0.0


@dataclass
class MCPSanitizeRequest:
    """Request to sanitize content."""
    content: str
    tool_name: Optional[str] = None
    request_id: str = ""


@dataclass
class MCPSanitizeResponse:
    """Response from sanitization."""
    request_id: str
    is_safe: bool
    sanitized_content: Optional[str] = None
    threats_detected: list[dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0


@dataclass
class MCPIntentRequest:
    """Request to verify intent alignment."""
    user_prompt: str
    proposed_actions: list[dict[str, str]]
    request_id: str = ""


@dataclass
class MCPIntentResponse:
    """Response from intent verification."""
    request_id: str
    is_aligned: bool
    similarity_scores: list[float] = field(default_factory=list)
    misaligned_actions: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


@dataclass
class MCPAnomalyRequest:
    """Request to detect anomalies in tool chain."""
    tool_sequence: list[str]
    request_id: str = ""


@dataclass
class MCPAnomalyResponse:
    """Response from anomaly detection."""
    request_id: str
    is_anomalous: bool
    anomaly_score: float = 0.0
    detected_pattern: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class MCPServerInfo:
    """Information about the ShieldAgent MCP server."""
    name: str = "shieldagent"
    version: str = "0.1.0"
    description: str = "Defense-in-depth security for agentic AI tool use"
    tools: list[str] = field(default_factory=lambda: [
        "sanitize_response",
        "verify_intent", 
        "detect_anomaly",
        "execute_sandboxed",
    ])
