"""Main ShieldAgent orchestrator - coordinates all defense modules."""

from typing import Any, Optional, Callable
import time
import asyncio
from dataclasses import dataclass, field
from loguru import logger

from shieldagent.core.config import ShieldAgentConfig
from shieldagent.core.types import (
    AgentResult,
    ThreatLevel,
    ThreatDetails,
    DefenseResult,
    ToolChain,
    ToolCall,
    UserIntent,
)
from shieldagent.defense.sanitizer import ToolResponseSanitizer
from shieldagent.defense.intent import IntentVerifier, ActionProposal
from shieldagent.defense.anomaly import ActionChainAnomalyDetector
from shieldagent.defense.sandbox import SandboxedExecutor
from shieldagent.llm import (
    get_llm_client,
    BaseLLMClient,
    ChatMessage,
    ToolDefinition,
)


@dataclass
class ProcessingContext:
    """Context for a single processing request."""
    user_intent: Optional[UserIntent] = None
    tool_chain: ToolChain = field(default_factory=ToolChain)
    defense_results: list[DefenseResult] = field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None


class ShieldAgent:
    """
    Main agent orchestrator with multi-layered defense.
    
    Defense layers (in order):
    1. Intent Verification - Check if actions align with user intent
    2. Tool Response Sanitization - Detect malicious content in tool outputs
    3. Action Chain Anomaly Detection - Detect suspicious tool sequences
    4. Sandboxed Execution - Isolate high-risk operations
    """
    
    def __init__(self, config: Optional[ShieldAgentConfig] = None):
        self.config = config or ShieldAgentConfig.from_env()
        
        # Initialize defense modules
        self.sanitizer = ToolResponseSanitizer(self.config.sanitizer)
        self.intent_verifier = IntentVerifier(self.config.intent_verifier)
        self.anomaly_detector = ActionChainAnomalyDetector(self.config.anomaly_detector)
        self.sandbox = SandboxedExecutor(self.config.sandbox)
        
        # LLM client (initialized on first use)
        self._llm_client: Optional[BaseLLMClient] = None
        
        # Metrics
        self._total_requests = 0
        self._blocked_requests = 0
        self._detected_attacks = 0
        
        logger.info("ShieldAgent initialized with defense layers:")
        logger.info(f"  - Sanitizer: {self.config.sanitizer.enabled}")
        logger.info(f"  - Intent Verification: {self.config.intent_verifier.enabled}")
        logger.info(f"  - Anomaly Detection: {self.config.anomaly_detector.enabled}")
        logger.info(f"  - Sandbox: {self.config.sandbox.enabled}")
    
    async def process(
        self,
        user_prompt: str,
        tools: Optional[list[ToolDefinition]] = None,
        messages: Optional[list[ChatMessage]] = None,
        tool_executor: Optional[Callable[[str, dict[str, Any]], Any] | dict[str, Callable[[dict[str, Any]], Any]]] = None,
    ) -> AgentResult:
        """
        Process a user request with full defense protection.
        
        Args:
            user_prompt: The user's prompt
            tools: Available tools for the agent
            messages: Optional previous messages for context
            
        Returns:
            AgentResult with output and security status
        """
        start_time = time.perf_counter()
        self._total_requests += 1
        
        context = ProcessingContext()
        
        # Step 1: Extract and encode user intent (only if enabled)
        if self.config.intent_verifier.enabled:
            context.user_intent = self.intent_verifier.extract_user_intent(user_prompt)
            logger.debug(f"Extracted intent goals: {context.user_intent.extracted_goals}")
        
        # Prepare messages
        conversation = list(messages) if messages else []
        conversation.append(ChatMessage(role="user", content=user_prompt))

        async def _maybe_await(result: Any) -> Any:
            if asyncio.iscoroutine(result):
                return await result
            return result

        def _get_tool_handler(tool_name: str) -> Optional[Callable[[dict[str, Any]], Any]]:
            if tool_executor is None:
                return None
            if isinstance(tool_executor, dict):
                return tool_executor.get(tool_name)
            if callable(tool_executor):
                return lambda args: tool_executor(tool_name, args)
            return None
        
        try:
            # Initialize LLM client if needed (uses factory for provider switching)
            if not self._llm_client:
                self._llm_client = get_llm_client(self.config.llm)
            
            async with self._llm_client as client:
                # Step 2: Get LLM response (may include tool calls)
                response = await client.chat(
                    messages=conversation,
                    tools=tools,
                    temperature=self.config.llm.temperature,
                )
                last_response = response
                tool_rounds = 0
                
                # Step 3: If tool calls requested, process with defense layers and loop
                while response.tool_calls and tool_executor is not None:
                    tool_rounds += 1
                    if tool_rounds > self.config.max_tool_rounds:
                        context.blocked = True
                        context.block_reason = "Tool call limit exceeded"
                        break
                    
                    # Append assistant message with tool call metadata before tool responses
                    conversation.append(ChatMessage(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=response.tool_calls,
                    ))
                    
                    for tc in response.tool_calls:
                        # Create tool call record
                        tool_call = ToolCall(
                            tool_name=tc.name,
                            arguments=tc.arguments,
                        )
                        context.tool_chain.add_call(tool_call)

                        # Check if tool is blocked (fail-fast)
                        if self.config.sandbox.enabled and self.sandbox.is_blocked(tc.name):
                            context.blocked = True
                            context.block_reason = f"Blocked tool: {tc.name}"
                            self._detected_attacks += 1
                            logger.warning(f"Blocked tool call: {tc.name}")
                            break
                        
                        # Verify intent alignment
                        if self.config.intent_verifier.enabled and context.user_intent:
                            action = ActionProposal(
                                action_type="tool_call",
                                description=f"Execute {tc.name} with {tc.arguments}",
                                tool_name=tc.name,
                                arguments=tc.arguments,
                            )
                            intent_result = self.intent_verifier.verify_action_chain(
                                context.user_intent,
                                [action],
                            )
                            context.defense_results.append(intent_result)
                            
                            if not intent_result.is_safe and self.config.fail_closed:
                                context.blocked = True
                                context.block_reason = "Intent verification failed"
                                break
                        
                        if context.blocked:
                            break
                        
                        # Optional tool execution + sandbox + sanitization
                        handler = _get_tool_handler(tc.name)
                        if not handler:
                            context.blocked = True
                            context.block_reason = f"No tool executor for: {tc.name}"
                            break
                        
                        if self.config.sandbox.enabled and self.sandbox.is_high_risk(tc.name):
                            logger.info(f"Executing high-risk tool {tc.name} in sandbox")
                            sandbox_result = await self.sandbox.execute_with_sandbox(
                                tool_call=tool_call,
                                executor_func=handler,
                            )
                            context.defense_results.append(sandbox_result)
                            
                            if not sandbox_result.is_safe and self.config.fail_closed:
                                context.blocked = True
                                context.block_reason = f"Sandbox blocked: {tc.name}"
                                break
                        else:
                            try:
                                tool_output = await _maybe_await(handler(tc.arguments))
                                tool_call.response = str(tool_output) if tool_output is not None else ""
                            except Exception as e:
                                logger.exception(f"Tool execution error for {tc.name}: {e}")
                                context.blocked = True
                                context.block_reason = f"Tool execution failed: {tc.name}"
                                break
                        
                        # Sanitize tool response before returning to LLM/caller
                        sanitizer_result = await self.process_tool_response(
                            tool_name=tc.name,
                            tool_response=tool_call.response or "",
                            context=context,
                        )
                        context.defense_results.append(sanitizer_result)
                        
                        if not sanitizer_result.is_safe and self.config.fail_closed:
                            context.blocked = True
                            context.block_reason = "Tool response sanitizer blocked content"
                            break
                        
                        tool_message_content = (
                            sanitizer_result.sanitized_content
                            if sanitizer_result.sanitized_content is not None
                            else (tool_call.response or "")
                        )
                        conversation.append(ChatMessage(
                            role="tool",
                            content=tool_message_content,
                            name=tc.name,
                            tool_call_id=tc.id,
                        ))
                    
                    if context.blocked:
                        break
                    
                    # Check action chain for anomalies before next round
                    if len(context.tool_chain) > 0:
                        anomaly_result = self.anomaly_detector.detect(context.tool_chain)
                        context.defense_results.append(anomaly_result)
                        
                        if not anomaly_result.is_safe and self.config.fail_closed:
                            context.blocked = True
                            context.block_reason = "Anomalous tool chain detected"
                            break
                    
                    # Request next response with tool results appended
                    response = await client.chat(
                        messages=conversation,
                        tools=tools,
                        temperature=self.config.llm.temperature,
                    )
                    last_response = response
                
                # Step 4: Block permanently forbidden tools (regardless of execution)
                if not context.blocked and last_response.tool_calls and self.config.sandbox.enabled:
                    for tc in last_response.tool_calls:
                        if self.sandbox.is_blocked(tc.name):
                            context.blocked = True
                            context.block_reason = f"Blocked tool: {tc.name}"
                            self._detected_attacks += 1
                            logger.warning(f"Blocked tool call: {tc.name}")
                            break
                
                # Build result
                total_latency = (time.perf_counter() - start_time) * 1000
                
                if context.blocked:
                    self._blocked_requests += 1
                    self._detected_attacks += 1
                    
                    # Find most severe threat
                    threats = [r.threat_detected for r in context.defense_results if r.threat_detected]
                    most_severe = max(threats, key=lambda t: t.threat_level.value) if threats else None
                    
                    return AgentResult(
                        success=False,
                        output=None,
                        attack_detected=True,
                        threat_details=most_severe,
                        defense_results=context.defense_results,
                        tool_chain=context.tool_chain,
                        total_latency_ms=total_latency,
                        blocked=True,
                        error=context.block_reason,
                    )
                
                return AgentResult(
                    success=True,
                    output=last_response.content if "last_response" in locals() else response.content,
                    attack_detected=False,
                    defense_results=context.defense_results,
                    tool_chain=context.tool_chain,
                    total_latency_ms=total_latency,
                )
                
        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                total_latency_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    async def process_tool_response(
        self,
        tool_name: str,
        tool_response: str,
        context: Optional[ProcessingContext] = None,
    ) -> DefenseResult:
        """
        Process a tool response through the sanitizer.
        
        This is called when a tool returns a response that needs
        to be checked before being passed back to the LLM.
        """
        if not self.config.sanitizer.enabled:
            return DefenseResult(
                module_name="sanitizer",
                is_safe=True,
                processing_time_ms=0.0,
                sanitized_content=tool_response,
            )
        sanitizer_result = self.sanitizer.sanitize(tool_response, tool_name)
        
        if sanitizer_result.threat_detected:
            self._detected_attacks += 1
            logger.warning(
                f"Threat in tool response from {tool_name}: "
                f"{sanitizer_result.threat_detected.attack_type.value}"
            )
        
        return sanitizer_result
    
    def get_metrics(self) -> dict:
        """Get agent performance metrics."""
        return {
            "total_requests": self._total_requests,
            "blocked_requests": self._blocked_requests,
            "detected_attacks": self._detected_attacks,
            "block_rate": self._blocked_requests / max(1, self._total_requests),
            "detection_rate": self._detected_attacks / max(1, self._total_requests),
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._total_requests = 0
        self._blocked_requests = 0
        self._detected_attacks = 0
    
    def shutdown(self) -> None:
        """Cleanup resources."""
        self.sandbox.shutdown()
        logger.info("ShieldAgent shutdown complete")


# Convenience function for quick usage
async def create_agent(
    model: str = "meta-llama/llama-3.1-70b-instruct",
    **kwargs,
) -> ShieldAgent:
    """Create a ShieldAgent with custom settings."""
    from shieldagent.core.config import LLMConfig
    
    config = ShieldAgentConfig.from_env()
    config.llm.model = model
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return ShieldAgent(config)
