"""Google AI (Gemini) LLM client for ShieldAgent."""

from typing import Any, AsyncIterator, Optional
import asyncio
import random
from loguru import logger

from shieldagent.core.config import LLMConfig
from shieldagent.llm.base import (
    BaseLLMClient,
    ChatMessage,
    ToolDefinition,
    ToolCallRequest,
    LLMResponse,
)


# Available Google AI models
GOOGLE_AI_MODELS = {
    # Short name -> full model ID
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-pro": "gemini-1.5-pro",
}


class GoogleAIClient(BaseLLMClient):
    """Client for Google AI (Gemini API) - Google's frontier models."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        
    async def __aenter__(self) -> "GoogleAIClient":
        try:
            from google import genai
            self._client = genai.Client(api_key=self.config.api_key)
        except ImportError:
            raise ImportError(
                "google-genai package required. Install with: pip install google-genai"
            )
        return self
    
    async def __aexit__(self, *args) -> None:
        self._client = None
    
    def _convert_messages_to_contents(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage format to Google AI contents format."""
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                # System message becomes system_instruction
                system_instruction = msg.content
            elif msg.role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == "tool":
                # Tool responses need special handling
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"Tool response: {msg.content}"}]
                })
        
        return contents, system_instruction
    
    def _convert_tools_to_google_format(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert ToolDefinition to Google AI tools format."""
        if not tools:
            return None
        
        function_declarations = []
        for tool in tools:
            function_declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        
        return [{"function_declarations": function_declarations}]
    
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict = "auto",
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request to Google AI."""
        
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        contents, system_instruction = self._convert_messages_to_contents(messages)
        
        # Build generation config
        generation_config = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        # Build request config
        config = {
            "generation_config": generation_config,
        }
        
        if system_instruction:
            config["system_instruction"] = system_instruction
        
        # Add tools if provided
        google_tools = self._convert_tools_to_google_format(tools)
        if google_tools:
            config["tools"] = google_tools
        
        logger.debug(f"Sending request to Google AI: {self.config.model}")
        
        # Retry with exponential backoff for rate limits
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Use async API
                response = await self._client.aio.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Google AI rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Google AI rate limit exceeded after {max_retries} retries")
                        raise
                else:
                    raise
        
        # Parse response
        tool_calls = None
        content = None
        
            # Detect function-call responses.
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        content = part.text
                    elif hasattr(part, 'function_call') and part.function_call:
                        if tool_calls is None:
                            tool_calls = []
                        fc = part.function_call
                        tool_calls.append(ToolCallRequest(
                            id=f"call_{len(tool_calls)}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        ))
        
            # Fallback path for plain-text responses.
        if content is None and hasattr(response, 'text'):
            content = response.text
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage=None,  # Google AI doesn't return usage the same way
        )
    
    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response from Google AI."""
        
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        contents, system_instruction = self._convert_messages_to_contents(messages)
        
        config = {
            "generation_config": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }
        
        if system_instruction:
            config["system_instruction"] = system_instruction
        
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self.config.model,
            contents=contents,
            config=config,
        ):
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
