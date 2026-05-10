"""ShieldAgent MCP Client with protection layer.

Wraps external MCP server connections with ShieldAgent security checks.
"""

import asyncio
from typing import Any, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from mcp import ClientSession, StdioServerParameters, stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not installed. Run: pip install mcp>=1.26.0")

from shieldagent.defense.sanitizer import ToolResponseSanitizer
from shieldagent.defense.intent import IntentVerifier, ActionProposal
from shieldagent.defense.anomaly import ActionChainAnomalyDetector
from shieldagent.core.config import (
    SanitizerConfig,
    IntentVerifierConfig,
    AnomalyDetectorConfig,
)
from shieldagent.core.types import ToolChain, ToolCall


@dataclass
class ProtectedToolResult:
    """Result from a protected tool call."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    was_blocked: bool = False
    block_reason: Optional[str] = None
    processing_time_ms: float = 0.0


class ShieldAgentMCPClient:
    """
    MCP Client wrapper that applies ShieldAgent protection to tool calls.
    
    All tool responses are sanitized before being returned to the caller.
    Tool chains are monitored for anomalous patterns.
    """
    
    def __init__(
        self,
        sanitizer_config: Optional[SanitizerConfig] = None,
        intent_config: Optional[IntentVerifierConfig] = None,
        anomaly_config: Optional[AnomalyDetectorConfig] = None,
        user_prompt: Optional[str] = None,
    ):
        """Initialize protected MCP client."""
        self.sanitizer = ToolResponseSanitizer(
            sanitizer_config or SanitizerConfig()
        )
        self.intent_verifier = IntentVerifier(
            intent_config or IntentVerifierConfig()
        )
        self.anomaly_detector = ActionChainAnomalyDetector(
            anomaly_config or AnomalyDetectorConfig()
        )
        
        # Track tool chain for anomaly detection
        self._tool_chain = ToolChain()
        
        # Store user intent if provided
        self._user_intent = None
        if user_prompt and self.intent_verifier.config.enabled:
            self._user_intent = self.intent_verifier.extract_user_intent(user_prompt)
        
        self._session: Optional[Any] = None
        
        logger.info("ShieldAgent protected MCP client initialized")
    
    def set_user_intent(self, prompt: str) -> None:
        """Set the user's intent for verification."""
        if self.intent_verifier.config.enabled:
            self._user_intent = self.intent_verifier.extract_user_intent(prompt)
        else:
            self._user_intent = None
    
    def reset_tool_chain(self) -> None:
        """Reset the tool chain tracking."""
        self._tool_chain = ToolChain()
    
    async def connect_stdio(
        self,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        """Connect to an MCP server via stdio."""
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP SDK not available")
        
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        
        self._stdio_context = stdio_client(params)
        read, write = await self._stdio_context.__aenter__()
        
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        
        logger.info(f"Connected to MCP server: {command}")
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if hasattr(self, '_stdio_context'):
            await self._stdio_context.__aexit__(None, None, None)
    
    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the connected server."""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")
        
        result = await self._session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.inputSchema,
            }
            for tool in result.tools
        ]
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        verify_intent: bool = True,
        check_anomaly: bool = True,
    ) -> ProtectedToolResult:
        """
        Call a tool with ShieldAgent protection.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            verify_intent: Whether to verify intent alignment
            check_anomaly: Whether to check for anomalous patterns
            
        Returns:
            ProtectedToolResult with sanitized output or block status
        """
        import time
        start_time = time.perf_counter()
        
        if not self._session:
            return ProtectedToolResult(
                success=False,
                error="Not connected to MCP server",
            )
        
        # Track this call
        call = ToolCall(tool_name=tool_name, arguments=arguments)
        self._tool_chain.add_call(call)
        
        # Check for anomalous tool chain patterns
        if check_anomaly and len(self._tool_chain) >= 2:
            anomaly_result = self.anomaly_detector.detect(self._tool_chain)
            if not anomaly_result.is_safe:
                logger.warning(f"Anomalous tool chain detected: {tool_name}")
                return ProtectedToolResult(
                    success=False,
                    was_blocked=True,
                    block_reason=anomaly_result.threat_detected.description if anomaly_result.threat_detected else "Anomalous pattern",
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                )
        
        # Verify intent alignment if we have user intent
        if verify_intent and self._user_intent:
            action = ActionProposal(
                action_type="tool_call",
                description=f"Execute {tool_name} with args: {list(arguments.keys())}",
                tool_name=tool_name,
                arguments=arguments,
            )
            is_aligned, _similarity = self.intent_verifier.verify_action(
                intent=self._user_intent,
                action=action,
            )
            if not is_aligned:
                logger.warning(f"Intent misalignment detected: {tool_name}")
                return ProtectedToolResult(
                    success=False,
                    was_blocked=True,
                    block_reason="Action does not align with user intent",
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                )
        
        # Execute the tool
        try:
            response = await self._session.call_tool(tool_name, arguments)
            raw_result = str(response.content[0].text) if response.content else ""
        except Exception as e:
            return ProtectedToolResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Sanitize the response
        sanitize_result = self.sanitizer.sanitize(raw_result, tool_name)
        
        if not sanitize_result.is_safe:
            logger.warning(f"Threat detected in tool response: {tool_name}")
            return ProtectedToolResult(
                success=False,
                was_blocked=True,
                block_reason=sanitize_result.threat_detected.description if sanitize_result.threat_detected else "Malicious content",
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        return ProtectedToolResult(
            success=True,
            result=sanitize_result.sanitized_content,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )


def create_protected_client(
    user_prompt: Optional[str] = None,
    sanitizer_config: Optional[SanitizerConfig] = None,
    intent_config: Optional[IntentVerifierConfig] = None,
    anomaly_config: Optional[AnomalyDetectorConfig] = None,
) -> ShieldAgentMCPClient:
    """Factory function to create protected MCP client."""
    return ShieldAgentMCPClient(
        user_prompt=user_prompt,
        sanitizer_config=sanitizer_config,
        intent_config=intent_config,
        anomaly_config=anomaly_config,
    )
