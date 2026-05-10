"""OpenRouter LLM client for ShieldAgent."""

from typing import Any, AsyncIterator, Optional
import asyncio
import random
import httpx
from loguru import logger

from shieldagent.core.config import LLMConfig
from shieldagent.llm.base import (
    BaseLLMClient,
    ChatMessage,
    ToolDefinition,
    ToolCallRequest,
    LLMResponse,
)


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API - unified access to multiple LLMs."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self) -> "OpenRouterClient":
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "HTTP-Referer": "https://github.com/shieldagent",
                "X-Title": "ShieldAgent Research",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )
        return self
    
    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
    
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict = "auto",
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request."""
        
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # Format messages
        formatted_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.name:
                m["name"] = msg.name
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                import json
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            formatted_messages.append(m)
        
        # Build request payload
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = tool_choice
        
        logger.debug(f"Sending request to OpenRouter: {self.config.model}")
        
        # Retry with exponential backoff for rate limits
        max_retries = 5
        base_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                response = await self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                break  # Success, exit retry loop
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limited (429), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} retries")
                        raise
                else:
                    raise  # Re-raise non-429 errors
        
        # Parse response
        choice = data["choices"][0]
        message = choice["message"]
        
        tool_calls = None
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = []
            for tc in message["tool_calls"]:
                import json
                tool_calls.append(ToolCallRequest(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                ))
        
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage"),
        )
    
    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response."""
        
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }
        
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    import json
                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]


# Available models on OpenRouter (subset)
AVAILABLE_MODELS = {
    # Meta Llama
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    
    # Mistral
    "mistral-7b": "mistralai/mistral-7b-instruct",
    "mistral-large": "mistralai/mistral-large",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
    
    # Anthropic
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    
    # OpenAI
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    
    # Google
    "gemini-pro": "google/gemini-pro",
    "gemini-2.0-flash": "google/gemini-2.0-flash-exp",
    
    # DeepSeek
    "deepseek-v3": "deepseek/deepseek-chat",
    "deepseek-coder": "deepseek/deepseek-coder",
    
    # Qwen
    "qwen-72b": "qwen/qwen-2.5-72b-instruct",
}


def get_model_id(short_name: str) -> str:
    """Get full model ID from short name."""
    return AVAILABLE_MODELS.get(short_name, short_name)
