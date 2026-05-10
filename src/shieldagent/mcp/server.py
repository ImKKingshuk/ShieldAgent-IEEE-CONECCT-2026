"""ShieldAgent MCP Server implementation.

Exposes ShieldAgent defense capabilities via the Model Context Protocol.
"""

import asyncio
import uuid
from typing import Any, Optional
from dataclasses import asdict
from loguru import logger

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not installed. Run: pip install mcp>=1.26.0")

from shieldagent.defense.sanitizer import ToolResponseSanitizer
from shieldagent.defense.intent import IntentVerifier
from shieldagent.defense.anomaly import ActionChainAnomalyDetector
from shieldagent.defense.sandbox import SandboxedExecutor
from shieldagent.core.config import (
    SanitizerConfig,
    IntentVerifierConfig,
    AnomalyDetectorConfig,
    SandboxConfig,
)
from shieldagent.core.types import (
    ToolChain,
    ToolCall,
    UserIntent,
)
from shieldagent.mcp.types import (
    MCPSanitizeRequest,
    MCPSanitizeResponse,
    MCPIntentRequest,
    MCPIntentResponse,
    MCPAnomalyRequest,
    MCPAnomalyResponse,
    MCPServerInfo,
)


class ShieldAgentMCPServer:
    """
    MCP Server that exposes ShieldAgent defense capabilities.
    
    Tools provided:
    - sanitize_response: Sanitize tool outputs for threats
    - verify_intent: Check if actions align with user intent
    - detect_anomaly: Analyze tool chains for suspicious patterns
    - execute_sandboxed: Execute tools in sandboxed environment
    """
    
    def __init__(
        self,
        sanitizer_config: Optional[SanitizerConfig] = None,
        intent_config: Optional[IntentVerifierConfig] = None,
        anomaly_config: Optional[AnomalyDetectorConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
    ):
        """Initialize ShieldAgent MCP Server with defense modules."""
        self.info = MCPServerInfo()
        
        # Initialize defense modules
        self.sanitizer = ToolResponseSanitizer(
            sanitizer_config or SanitizerConfig()
        )
        self.intent_verifier = IntentVerifier(
            intent_config or IntentVerifierConfig()
        )
        self.anomaly_detector = ActionChainAnomalyDetector(
            anomaly_config or AnomalyDetectorConfig()
        )
        self.sandbox = SandboxedExecutor(
            sandbox_config or SandboxConfig()
        )
        
        # Initialize MCP server if available
        self._mcp_server: Optional[Any] = None
        if MCP_AVAILABLE:
            self._setup_mcp_server()
        
        logger.info(f"ShieldAgent MCP Server initialized with tools: {self.info.tools}")
    
    def _setup_mcp_server(self) -> None:
        """Set up the MCP server with tool definitions."""
        self._mcp_server = Server(self.info.name)
        
        @self._mcp_server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="sanitize_response",
                    description="Sanitize tool response content for malicious patterns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Content to sanitize"},
                            "tool_name": {"type": "string", "description": "Optional tool name for context"},
                        },
                        "required": ["content"],
                    },
                ),
                Tool(
                    name="verify_intent",
                    description="Verify if proposed actions align with user intent",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_prompt": {"type": "string", "description": "Original user prompt"},
                            "proposed_actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "tool": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                },
                                "description": "List of proposed actions",
                            },
                        },
                        "required": ["user_prompt", "proposed_actions"],
                    },
                ),
                Tool(
                    name="detect_anomaly",
                    description="Detect anomalous patterns in tool call sequences",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tool_sequence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Sequence of tool names",
                            },
                        },
                        "required": ["tool_sequence"],
                    },
                ),
                Tool(
                    name="execute_sandboxed",
                    description="Execute a tool call in sandboxed environment",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "Tool to execute"},
                            "arguments": {"type": "object", "description": "Tool arguments"},
                        },
                        "required": ["tool_name", "arguments"],
                    },
                ),
            ]
        
        @self._mcp_server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            result = await self._handle_tool_call(name, arguments)
            return [TextContent(type="text", text=str(result))]
    
    async def _handle_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Route tool calls to appropriate handlers."""
        request_id = str(uuid.uuid4())[:8]
        
        handlers = {
            "sanitize_response": self._handle_sanitize,
            "verify_intent": self._handle_verify_intent,
            "detect_anomaly": self._handle_detect_anomaly,
            "execute_sandboxed": self._handle_execute_sandboxed,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}", "request_id": request_id}
        
        try:
            return await handler(request_id, arguments)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e), "request_id": request_id}
    
    async def _handle_sanitize(
        self,
        request_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle sanitize_response tool calls."""
        content = arguments.get("content", "")
        tool_name = arguments.get("tool_name")
        
        result = self.sanitizer.sanitize(content, tool_name)
        
        response = MCPSanitizeResponse(
            request_id=request_id,
            is_safe=result.is_safe,
            sanitized_content=result.sanitized_content,
            threats_detected=[
                asdict(result.threat_detected)
            ] if result.threat_detected else [],
            processing_time_ms=result.processing_time_ms,
        )
        
        return asdict(response)
    
    async def _handle_verify_intent(
        self,
        request_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle verify_intent tool calls."""
        user_prompt = arguments.get("user_prompt", "")
        proposed_actions = arguments.get("proposed_actions", [])
        
        if not self.intent_verifier.config.enabled:
            response = MCPIntentResponse(
                request_id=request_id,
                is_aligned=True,
                similarity_scores=[],
                misaligned_actions=[],
                processing_time_ms=0.0,
            )
            return asdict(response)

        # Extract user intent
        intent = self.intent_verifier.extract_user_intent(user_prompt)
        
        # Verify each action
        similarity_scores = []
        misaligned_actions = []
        
        for action in proposed_actions:
            action_desc = action.get("description", action.get("tool", ""))
            similarity = self.intent_verifier.compute_similarity(intent, action_desc)
            similarity_scores.append(similarity)
            
            if similarity < self.intent_verifier.config.similarity_threshold:
                misaligned_actions.append(action.get("tool", "unknown"))
        
        response = MCPIntentResponse(
            request_id=request_id,
            is_aligned=len(misaligned_actions) == 0,
            similarity_scores=similarity_scores,
            misaligned_actions=misaligned_actions,
            processing_time_ms=0.0,
        )
        
        return asdict(response)
    
    async def _handle_detect_anomaly(
        self,
        request_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle detect_anomaly tool calls."""
        tool_sequence = arguments.get("tool_sequence", [])
        
        # Create tool chain
        chain = ToolChain()
        for tool_name in tool_sequence:
            chain.add_call(ToolCall(tool_name=tool_name, arguments={}))
        
        result = self.anomaly_detector.detect(chain)
        
        response = MCPAnomalyResponse(
            request_id=request_id,
            is_anomalous=not result.is_safe,
            anomaly_score=result.threat_detected.confidence if result.threat_detected else 0.0,
            detected_pattern=result.threat_detected.description if result.threat_detected else None,
            processing_time_ms=result.processing_time_ms,
        )
        
        return asdict(response)
    
    async def _handle_execute_sandboxed(
        self,
        request_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle execute_sandboxed tool calls."""
        tool_name = arguments.get("tool_name", "")
        tool_args = arguments.get("arguments", {})
        
        # Check if tool is blocked
        if self.sandbox.is_blocked(tool_name):
            return {
                "request_id": request_id,
                "success": False,
                "error": f"Tool '{tool_name}' is blocked",
                "is_safe": False,
            }
        
        # For security, we don't actually execute unknown tools
        # Return sandbox check result instead
        return {
            "request_id": request_id,
            "success": True,
            "is_high_risk": self.sandbox.is_high_risk(tool_name),
            "is_blocked": False,
            "message": f"Tool '{tool_name}' cleared for sandbox execution",
        }
    
    def get_mcp_server(self) -> Optional[Any]:
        """Get the underlying MCP server instance."""
        return self._mcp_server
    
    async def run_stdio(self) -> None:
        """Run the MCP server using stdio transport."""
        if not MCP_AVAILABLE or not self._mcp_server:
            raise RuntimeError("MCP SDK not available")
        
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self._mcp_server.run(
                read_stream,
                write_stream,
                self._mcp_server.create_initialization_options(),
            )


def create_server(
    sanitizer_config: Optional[SanitizerConfig] = None,
    intent_config: Optional[IntentVerifierConfig] = None,
    anomaly_config: Optional[AnomalyDetectorConfig] = None,
    sandbox_config: Optional[SandboxConfig] = None,
) -> ShieldAgentMCPServer:
    """Factory function to create ShieldAgent MCP Server."""
    return ShieldAgentMCPServer(
        sanitizer_config=sanitizer_config,
        intent_config=intent_config,
        anomaly_config=anomaly_config,
        sandbox_config=sandbox_config,
    )


async def main() -> None:
    """Main entry point for running MCP server."""
    server = create_server()
    logger.info("Starting ShieldAgent MCP Server...")
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
