"""Cerebras LLM client for ShieldAgent - uses OpenAI-compatible API."""

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


# Available Cerebras models
CEREBRAS_MODELS = {
    # Short name -> full model ID
    "llama-3.3-70b": "llama-3.3-70b",
    "llama3.1-8b": "llama3.1-8b", 
    "qwen-3-32b": "qwen-3-32b",
}


class CerebrasClient(BaseLLMClient):
    """Client for Cerebras API - fast inference on Wafer-Scale Engine."""
    
    BASE_URL = "https://api.cerebras.ai/v1"
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self) -> "CerebrasClient":
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
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
        """Send a chat completion request to Cerebras."""
        
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
        
        # Add tools if provided (Cerebras supports function calling)
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
        
        logger.debug(f"Sending request to Cerebras: {self.config.model}")
        
        # Retry with exponential backoff for rate limits
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = await self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Cerebras rate limited (429), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Cerebras rate limit exceeded after {max_retries} retries")
                        raise
                else:
                    raise
        
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
        """Stream a chat completion response from Cerebras."""
        
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
                    try:
                        import json
                        chunk = json.loads(data)
                        if chunk["choices"][0]["delta"].get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
