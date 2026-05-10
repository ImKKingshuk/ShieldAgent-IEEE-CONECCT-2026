"""Sandboxed Execution Engine for high-risk tool operations."""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from loguru import logger

from shieldagent.core.types import (
    ThreatLevel,
    AttackType,
    ThreatDetails,
    DefenseResult,
    ToolCall,
)
from shieldagent.core.config import SandboxConfig


@dataclass
class SandboxState:
    """Tracks the state of sandboxed execution for rollback."""
    created_files: list[str] = field(default_factory=list)
    modified_files: dict[str, bytes] = field(default_factory=dict)  # path -> original content
    network_requests: list[dict] = field(default_factory=list)
    executed_commands: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    state: Optional[SandboxState] = None


# High-risk tools that require sandboxing
HIGH_RISK_TOOLS = {
    "shell_execute",
    "code_execute",
    "file_write",
    "file_delete",
    "db_delete",
    "db_update",
    "process_kill",
    "env_write",
    "system_modify",
}

# Tools that should never be executed
BLOCKED_TOOLS = {
    "root_access",
    "kernel_modify",
    "admin_override",
}


class SandboxedExecutor:
    """
    Provides sandboxed execution environment for high-risk operations.
    
    Features:
    - Timeout enforcement
    - State tracking for rollback
    - Resource limits
    - Network isolation (optional)
    """
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._current_state: Optional[SandboxState] = None
    
    def is_high_risk(self, tool_name: str) -> bool:
        """Check if a tool is considered high-risk."""
        return tool_name.lower() in HIGH_RISK_TOOLS
    
    def is_blocked(self, tool_name: str) -> bool:
        """Check if a tool is blocked entirely."""
        return tool_name.lower() in BLOCKED_TOOLS
    
    def _create_sandbox_state(self) -> SandboxState:
        """Create a new sandbox state for tracking changes."""
        return SandboxState()
    
    async def execute_with_sandbox(
        self,
        tool_call: ToolCall,
        executor_func: Callable[..., Any],
    ) -> DefenseResult:
        """
        Execute a tool call within sandbox constraints.
        
        Args:
            tool_call: The tool call to execute
            executor_func: Function to actually execute the tool
            
        Returns:
            DefenseResult with execution outcome
        """
        start_time = time.perf_counter()
        
        if not self.config.enabled:
            # No sandbox - direct execution
            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        executor_func,
                        tool_call.arguments,
                    ),
                    timeout=self.config.timeout_seconds,
                )
                tool_call.response = str(result)
                return DefenseResult(
                    module_name="sandbox",
                    is_safe=True,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return DefenseResult(
                    module_name="sandbox",
                    is_safe=False,
                    threat_detected=ThreatDetails(
                        attack_type=AttackType.UNKNOWN,
                        threat_level=ThreatLevel.MEDIUM,
                        description=f"Tool execution error: {str(e)}",
                        confidence=0.5,
                    ),
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                )
        
        # Check if tool is blocked
        if self.is_blocked(tool_call.tool_name):
            threat = ThreatDetails(
                attack_type=AttackType.PRIVILEGE_ESCALATION,
                threat_level=ThreatLevel.CRITICAL,
                description=f"Blocked tool attempted: {tool_call.tool_name}",
                evidence=[f"Tool '{tool_call.tool_name}' is permanently blocked"],
                confidence=1.0,
                affected_tools=[tool_call.tool_name],
                mitigation_applied="execution_blocked",
            )
            
            logger.warning(f"Blocked tool execution attempted: {tool_call.tool_name}")
            
            return DefenseResult(
                module_name="sandbox",
                is_safe=False,
                threat_detected=threat,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Create sandbox state
        self._current_state = self._create_sandbox_state()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._sandboxed_execute(tool_call, executor_func),
                timeout=self.config.timeout_seconds,
            )
            
            tool_call.response = str(result.output) if result.output else None
            tool_call.execution_time_ms = result.execution_time_ms
            
            if not result.success and self.config.rollback_on_error:
                await self._rollback(self._current_state)
            
            return DefenseResult(
                module_name="sandbox",
                is_safe=result.success,
                threat_detected=ThreatDetails(
                    attack_type=AttackType.UNKNOWN,
                    threat_level=ThreatLevel.LOW,
                    description=result.error or "Unknown error",
                    confidence=0.6,
                ) if not result.success else None,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Tool execution timed out: {tool_call.tool_name}")
            
            if self.config.rollback_on_error:
                await self._rollback(self._current_state)
            
            return DefenseResult(
                module_name="sandbox",
                is_safe=False,
                threat_detected=ThreatDetails(
                    attack_type=AttackType.UNKNOWN,
                    threat_level=ThreatLevel.HIGH,
                    description=f"Tool execution timed out after {self.config.timeout_seconds}s",
                    evidence=[f"Tool '{tool_call.tool_name}' exceeded time limit"],
                    confidence=0.8,
                    affected_tools=[tool_call.tool_name],
                    mitigation_applied="execution_killed",
                ),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            
            if self.config.rollback_on_error:
                await self._rollback(self._current_state)
            
            return DefenseResult(
                module_name="sandbox",
                is_safe=False,
                threat_detected=ThreatDetails(
                    attack_type=AttackType.UNKNOWN,
                    threat_level=ThreatLevel.MEDIUM,
                    description=f"Sandbox execution error: {str(e)}",
                    confidence=0.6,
                ),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    async def _sandboxed_execute(
        self,
        tool_call: ToolCall,
        executor_func: Callable[..., Any],
    ) -> ExecutionResult:
        """
        Execute tool within sandbox constraints.
        
        This is a simplified sandbox - in production, would use:
        - Docker containers with resource limits
        - seccomp filters
        - Network namespaces
        - gVisor or similar
        """
        start_time = time.perf_counter()
        state = self._current_state or SandboxState()
        
        try:
            # Track high-risk operations
            if tool_call.tool_name == "file_write":
                path = tool_call.arguments.get("path", "")
                if path:
                    import os
                    if os.path.exists(path):
                        try:
                            with open(path, "rb") as f:
                                state.modified_files[path] = f.read()
                        except Exception as e:
                            logger.warning(f"Failed to snapshot file before write {path}: {e}")
                    else:
                        state.created_files.append(path)
            
            elif tool_call.tool_name == "shell_execute":
                cmd = tool_call.arguments.get("command", "")
                state.executed_commands.append(cmd)
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                executor_func,
                tool_call.arguments,
            )
            
            return ExecutionResult(
                success=True,
                output=result,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                state=state,
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                state=state,
            )
    
    async def _rollback(self, state: SandboxState) -> None:
        """Rollback changes tracked in sandbox state."""
        logger.info("Rolling back sandbox changes...")
        
        # Remove created files
        for path in state.created_files:
            try:
                import os
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed created file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove file {path}: {e}")
        
        # Restore modified files
        for path, original_content in state.modified_files.items():
            try:
                with open(path, 'wb') as f:
                    f.write(original_content)
                logger.debug(f"Restored file: {path}")
            except Exception as e:
                logger.warning(f"Failed to restore file {path}: {e}")
        
        logger.info("Rollback completed")
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
