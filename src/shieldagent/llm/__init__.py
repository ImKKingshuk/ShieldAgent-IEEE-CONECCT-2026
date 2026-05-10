"""LLM client implementations for ShieldAgent.

Supports multiple providers:
- OpenRouter: Unified access to 200+ models (OpenAI, Anthropic, Meta, etc.)
- Cerebras: Ultra-fast inference on Wafer-Scale Engine
- Google AI: Gemini models (1.5, 2.0, 2.5 Flash/Pro)

Usage:
    from shieldagent.llm import get_llm_client, ChatMessage, ToolDefinition
    
    config = LLMConfig(provider="openrouter", model="meta-llama/llama-3.1-70b-instruct")
    async with get_llm_client(config) as client:
        response = await client.chat([ChatMessage(role="user", content="Hello!")])
"""

from shieldagent.llm.base import (
    BaseLLMClient,
    ChatMessage,
    ToolDefinition,
    ToolCallRequest,
    LLMResponse,
)
from shieldagent.llm.openrouter import OpenRouterClient, AVAILABLE_MODELS
from shieldagent.llm.cerebras import CerebrasClient, CEREBRAS_MODELS
from shieldagent.llm.google_ai import GoogleAIClient, GOOGLE_AI_MODELS
from shieldagent.core.config import LLMConfig


def get_llm_client(config: LLMConfig) -> BaseLLMClient:
    """
    Factory function to get the appropriate LLM client based on config.
    
    Args:
        config: LLMConfig with provider and model settings
        
    Returns:
        BaseLLMClient instance for the specified provider
        
    Raises:
        ValueError: If provider is not supported
        
    Supported providers:
        - "openrouter": OpenRouter API (200+ models)
        - "cerebras": Cerebras Cloud (fast Llama/Qwen inference)
        - "google", "google_ai", "gemini": Google AI (Gemini models)
    """
    provider = config.provider.lower()
    
    if provider == "openrouter":
        return OpenRouterClient(config)
    elif provider == "cerebras":
        return CerebrasClient(config)
    elif provider in ("google", "google_ai", "gemini"):
        return GoogleAIClient(config)
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported: openrouter, cerebras, google"
        )


# All available models by provider
ALL_MODELS = {
    "openrouter": AVAILABLE_MODELS,
    "cerebras": CEREBRAS_MODELS,
    "google": GOOGLE_AI_MODELS,
}


__all__ = [
    # Factory
    "get_llm_client",
    # Base types
    "BaseLLMClient",
    "ChatMessage",
    "ToolDefinition", 
    "ToolCallRequest",
    "LLMResponse",
    # Clients
    "OpenRouterClient",
    "CerebrasClient",
    "GoogleAIClient",
    # Model lists
    "AVAILABLE_MODELS",
    "CEREBRAS_MODELS",
    "GOOGLE_AI_MODELS",
    "ALL_MODELS",
]
