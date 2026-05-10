"""MCP Server wrapper for ShieldAgent.

Provides Model Context Protocol integration for ShieldAgent defense capabilities.
"""

from shieldagent.mcp.types import (
    MCPMessageType,
    MCPToolRequest,
    MCPToolResponse,
    MCPSanitizeRequest,
    MCPSanitizeResponse,
    MCPIntentRequest,
    MCPIntentResponse,
    MCPAnomalyRequest,
    MCPAnomalyResponse,
    MCPServerInfo,
)

from shieldagent.mcp.server import (
    ShieldAgentMCPServer,
    create_server,
)

from shieldagent.mcp.client import (
    ShieldAgentMCPClient,
    ProtectedToolResult,
    create_protected_client,
)


__all__ = [
    # Types
    "MCPMessageType",
    "MCPToolRequest",
    "MCPToolResponse",
    "MCPSanitizeRequest",
    "MCPSanitizeResponse",
    "MCPIntentRequest",
    "MCPIntentResponse",
    "MCPAnomalyRequest",
    "MCPAnomalyResponse",
    "MCPServerInfo",
    # Server
    "ShieldAgentMCPServer",
    "create_server",
    # Client
    "ShieldAgentMCPClient",
    "ProtectedToolResult",
    "create_protected_client",
]
