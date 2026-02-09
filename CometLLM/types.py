"""类型定义和数据模型."""
from __future__ import annotations
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
from pydantic import BaseModel, Field

class Provider(str, Enum):
    """支持的LLM提供商."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class Role(str, Enum):
    """消息角色."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Message(BaseModel):
    """聊天消息."""
    role: Role
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class Usage(BaseModel):
    """Token使用量."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class Choice(BaseModel):
    """完成选项."""
    index: int = 0
    message: Message
    finish_reason: Optional[str] = None

class ChoiceDelta(BaseModel):
    """流式响应增量."""
    index: int = 0
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletion(BaseModel):
    """聊天完成响应."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None

class ChatCompletionChunk(BaseModel):
    """流式聊天完成响应块."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChoiceDelta]
    usage: Optional[Usage] = None

class ToolFunction(BaseModel):
    """工具函数定义."""
    name: str
    description: str
    parameters: Dict[str, Any]

class Tool(BaseModel):
    """工具定义."""
    type: str = "function"
    function: ToolFunction

class LLMConfig(BaseModel):
    """LLM配置."""
    provider: Provider
    api_key: str
    base_url: Optional[str] = None
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0.0)
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class StreamOptions(BaseModel):
    """流式选项."""
    include_usage: bool = False

ChatCompletionResponse = Union[ChatCompletion, Iterator[ChatCompletionChunk]]
AsyncChatCompletionResponse = Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]