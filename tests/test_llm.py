"""Unit tests for OpenRouter LLM client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from shieldagent.llm.openrouter import (
    OpenRouterClient,
    ChatMessage,
    ToolDefinition,
    ToolCallRequest,
    LLMResponse,
    AVAILABLE_MODELS,
    get_model_id,
)
from shieldagent.core.config import LLMConfig


@pytest.fixture
def llm_config():
    """Create test LLM config."""
    return LLMConfig(
        api_key="test-api-key",
        model="meta-llama/llama-3.1-70b-instruct",
        temperature=0.7,
        max_tokens=4096,
        timeout=30.0,
    )


@pytest.fixture
def client(llm_config):
    """Create test client."""
    return OpenRouterClient(llm_config)


class TestChatMessage:
    """Tests for ChatMessage dataclass."""
    
    def test_basic_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_call_id is None
    
    def test_tool_message(self):
        msg = ChatMessage(
            role="tool",
            content="Tool output",
            name="file_read",
            tool_call_id="call_123",
        )
        assert msg.role == "tool"
        assert msg.name == "file_read"
        assert msg.tool_call_id == "call_123"


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""
    
    def test_tool_definition(self):
        tool = ToolDefinition(
            name="file_read",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        assert tool.name == "file_read"
        assert tool.description == "Read a file"
        assert "path" in tool.parameters["properties"]


class TestToolCallRequest:
    """Tests for ToolCallRequest dataclass."""
    
    def test_tool_call_request(self):
        tc = ToolCallRequest(
            id="call_abc",
            name="shell_execute",
            arguments={"command": "ls -la"},
        )
        assert tc.id == "call_abc"
        assert tc.name == "shell_execute"
        assert tc.arguments["command"] == "ls -la"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""
    
    def test_text_response(self):
        resp = LLMResponse(content="Hello there!")
        assert resp.content == "Hello there!"
        assert resp.tool_calls is None
        assert resp.finish_reason == "stop"
    
    def test_tool_call_response(self):
        tc = ToolCallRequest(id="1", name="test", arguments={})
        resp = LLMResponse(tool_calls=[tc], finish_reason="tool_calls")
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1


class TestOpenRouterClient:
    """Tests for OpenRouterClient."""
    
    def test_initialization(self, llm_config):
        client = OpenRouterClient(llm_config)
        assert client.config == llm_config
        assert client._client is None
    
    @pytest.mark.asyncio
    async def test_context_manager_enters(self, client):
        async with client as c:
            assert c._client is not None
            assert isinstance(c._client, httpx.AsyncClient)
    
    @pytest.mark.asyncio
    async def test_context_manager_exits(self, client):
        async with client:
            pass
        # After exit, client should be closed
        # (internal state is not exposed, but shouldn't raise)
    
    @pytest.mark.asyncio
    async def test_chat_without_context_raises(self, client):
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(RuntimeError, match="not initialized"):
            await client.chat(messages)
    
    @pytest.mark.asyncio
    async def test_chat_formats_messages_correctly(self, client, llm_config):
        """Test that messages are formatted correctly for API."""
        mock_response = {
            "choices": [{
                "message": {"content": "Hi!", "role": "assistant"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        
        with patch.object(client, '_client', new_callable=MagicMock) as mock_client:
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    json=lambda: mock_response,
                    raise_for_status=lambda: None,
                )
            )
            
            messages = [
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ]
            
            response = await client.chat(messages)
            
            # Verify response
            assert response.content == "Hi!"
            assert response.finish_reason == "stop"
            
            # Verify API call was made
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "/chat/completions"
    
    @pytest.mark.asyncio
    async def test_chat_with_tools(self, client):
        """Test that tools are formatted correctly."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": '{"path": "/tmp/test.txt"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        
        with patch.object(client, '_client', new_callable=MagicMock) as mock_client:
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    json=lambda: mock_response,
                    raise_for_status=lambda: None,
                )
            )
            
            tools = [
                ToolDefinition(
                    name="file_read",
                    description="Read a file",
                    parameters={"type": "object", "properties": {}},
                )
            ]
            
            response = await client.chat(
                messages=[ChatMessage(role="user", content="Read the file")],
                tools=tools,
            )
            
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "file_read"
            assert response.tool_calls[0].arguments == {"path": "/tmp/test.txt"}
    
    @pytest.mark.asyncio
    async def test_stream_without_context_raises(self, client):
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in client.stream(messages):
                pass


class TestAvailableModels:
    """Tests for model mapping."""
    
    def test_available_models_not_empty(self):
        assert len(AVAILABLE_MODELS) > 0
    
    def test_llama_models_available(self):
        assert "llama-3.1-70b" in AVAILABLE_MODELS
        assert "llama-3.1-8b" in AVAILABLE_MODELS
    
    def test_get_model_id_short_name(self):
        model_id = get_model_id("llama-3.1-70b")
        assert model_id == "meta-llama/llama-3.1-70b-instruct"
    
    def test_get_model_id_full_name(self):
        # Returns full name as-is if not in map
        model_id = get_model_id("openai/gpt-4-turbo")
        assert model_id == "openai/gpt-4-turbo"
    
    def test_all_model_ids_have_slash(self):
        """All full model IDs should have org/model format."""
        for full_id in AVAILABLE_MODELS.values():
            assert "/" in full_id, f"Model {full_id} should have org/model format"


class TestClientHeaders:
    """Tests for client header configuration."""
    
    @pytest.mark.asyncio
    async def test_headers_configured(self, client):
        async with client as c:
            headers = c._client.headers
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-api-key"
            assert headers["Content-Type"] == "application/json"
