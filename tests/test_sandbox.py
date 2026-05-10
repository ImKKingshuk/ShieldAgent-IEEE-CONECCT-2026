"""Unit tests for the Sandboxed Execution Engine."""

import pytest
from unittest.mock import MagicMock, patch
from shieldagent.defense.sandbox import (
    SandboxedExecutor,
    HIGH_RISK_TOOLS,
    BLOCKED_TOOLS,
    SandboxState,
)
from shieldagent.core.config import SandboxConfig
from shieldagent.core.types import ThreatLevel, ToolCall


@pytest.fixture
def sandbox():
    """Create a sandbox executor with default config."""
    config = SandboxConfig(
        enabled=True,
        use_docker=False,
        timeout_seconds=5.0,
        max_memory_mb=256,
        network_disabled=True,
        rollback_on_error=True,
    )
    return SandboxedExecutor(config)


class TestToolClassification:
    """Tests for tool risk classification."""
    
    def test_high_risk_tools_defined(self):
        """Should have defined high-risk tools."""
        assert len(HIGH_RISK_TOOLS) > 0
        assert "shell_execute" in HIGH_RISK_TOOLS
        assert "code_execute" in HIGH_RISK_TOOLS
        assert "file_delete" in HIGH_RISK_TOOLS
    
    def test_blocked_tools_defined(self):
        """Should have defined blocked tools."""
        assert len(BLOCKED_TOOLS) > 0
        assert "root_access" in BLOCKED_TOOLS
        assert "kernel_modify" in BLOCKED_TOOLS
    
    def test_is_high_risk(self, sandbox):
        """Should correctly identify high-risk tools."""
        assert sandbox.is_high_risk("shell_execute")
        assert sandbox.is_high_risk("file_write")
        assert not sandbox.is_high_risk("file_read")
        assert not sandbox.is_high_risk("web_search")
    
    def test_is_blocked(self, sandbox):
        """Should correctly identify blocked tools."""
        assert sandbox.is_blocked("root_access")
        assert sandbox.is_blocked("kernel_modify")
        assert not sandbox.is_blocked("shell_execute")
        assert not sandbox.is_blocked("file_read")


class TestBlockedToolExecution:
    """Tests for blocked tool handling."""
    
    @pytest.mark.asyncio
    async def test_blocked_tool_rejected(self, sandbox):
        """Should reject execution of blocked tools."""
        tool_call = ToolCall(tool_name="root_access", arguments={"command": "whoami"})
        
        result = await sandbox.execute_with_sandbox(
            tool_call=tool_call,
            executor_func=lambda tc: "executed",
        )
        
        assert not result.is_safe
        assert result.threat_detected is not None
        assert result.threat_detected.threat_level == ThreatLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_blocked_tool_no_execution(self, sandbox):
        """Should not execute blocked tool handlers."""
        executed = False
        
        def handler(tc):
            nonlocal executed
            executed = True
            return "result"
        
        tool_call = ToolCall(tool_name="kernel_modify", arguments={})
        
        await sandbox.execute_with_sandbox(
            tool_call=tool_call,
            executor_func=handler,
        )
        
        assert not executed


class TestHighRiskToolExecution:
    """Tests for high-risk tool handling."""
    
    @pytest.mark.asyncio
    async def test_high_risk_tool_executed_with_sandbox(self, sandbox):
        """Should execute high-risk tools within sandbox."""
        tool_call = ToolCall(
            tool_name="shell_execute",
            arguments={"command": "echo test"}
        )
        
        result = await sandbox.execute_with_sandbox(
            tool_call=tool_call,
            executor_func=lambda tc: "test output",
        )
        
        # For successful sandbox executions, result is safe
        # (sanitized_content is None because no sanitization occurred)
        assert result.is_safe


class TestSafeToolExecution:
    """Tests for safe tool execution."""
    
    @pytest.mark.asyncio
    async def test_safe_tool_passes(self, sandbox):
        """Should execute safe tools without restrictions."""
        tool_call = ToolCall(
            tool_name="file_read",
            arguments={"path": "/tmp/test.txt"}
        )
        
        result = await sandbox.execute_with_sandbox(
            tool_call=tool_call,
            executor_func=lambda tc: "file contents",
        )
        
        # For successful safe tool executions, result is safe
        # (sanitized_content is None because no sanitization occurred)
        assert result.is_safe
        assert result.threat_detected is None
    
    @pytest.mark.asyncio
    async def test_handler_error_caught(self, sandbox):
        """Should catch and handle errors from handlers."""
        def error_handler(tc):
            raise ValueError("Handler error")
        
        tool_call = ToolCall(tool_name="file_read", arguments={})
        
        result = await sandbox.execute_with_sandbox(
            tool_call=tool_call,
            executor_func=error_handler,
        )
        
        assert not result.is_safe
        assert result.threat_detected is not None


class TestStateManagement:
    """Tests for execution state tracking."""
    
    def test_create_sandbox_state(self, sandbox):
        """Should create sandbox state."""
        state = sandbox._create_sandbox_state()
        
        assert state is not None
        assert isinstance(state, SandboxState)
        assert state.created_files == []
        assert state.modified_files == {}


class TestDisabledSandbox:
    """Tests for disabled sandbox mode."""
    
    @pytest.fixture
    def disabled_sandbox(self):
        """Create a disabled sandbox."""
        config = SandboxConfig(enabled=False)
        return SandboxedExecutor(config)
    
    @pytest.mark.asyncio
    async def test_disabled_passes_all(self, disabled_sandbox):
        """Should pass all tools when disabled."""
        tool_call = ToolCall(tool_name="root_access", arguments={})
        
        result = await disabled_sandbox.execute_with_sandbox(
            tool_call=tool_call,
            executor_func=lambda tc: "executed",
        )
        
        # When sandbox is disabled, everything passes through
        assert result.is_safe


class TestShutdown:
    """Tests for resource cleanup."""
    
    def test_shutdown_cleans_executor(self, sandbox):
        """Should cleanup thread pool on shutdown."""
        sandbox.shutdown()
        
        # Executor should be shutdown
        assert sandbox._executor._shutdown
