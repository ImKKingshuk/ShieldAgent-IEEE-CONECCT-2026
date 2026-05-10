"""Base LLM client interface and common types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional


@dataclass
class ChatMessage:
    """A chat message."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: list["ToolCallRequest"] | None = None


@dataclass
class ToolDefinition:
    """Definition of a tool for function calling."""
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCallRequest:
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass 
class LLMResponse:
    """Response from the LLM."""
    content: Optional[str] = None
    tool_calls: list[ToolCallRequest] | None = None
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def __aenter__(self) -> "BaseLLMClient":
        """Enter async context."""
        pass
    
    @abstractmethod
    async def __aexit__(self, *args) -> None:
        """Exit async context."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict = "auto",
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request."""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response."""
        pass
