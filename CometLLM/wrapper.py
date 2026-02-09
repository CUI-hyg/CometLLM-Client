"""CometLLM 高级包装器"""
from __future__ import annotations
import os
import asyncio
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union, overload
from dataclasses import dataclass, field
from contextlib import contextmanager
from dotenv import load_dotenv
from . import CometLLM, Message, ChatCompletion, ChatCompletionChunk
from .types import Tool, Usage
from .exceptions import CometLLMError

@dataclass
class ChatMessage:
    """简化版聊天消息."""
    role: str
    content: str

@dataclass
class ChatResponse:
    """简化版聊天响应."""
    content: str
    usage: Optional[Usage] = None
    raw_response: Optional[ChatCompletion] = None

@dataclass
class ChatSession:
    """对话会话，自动维护历史记录."""
    messages: List[Message] = field(default_factory=list)
    max_history: int = 20
    
    def add_user(self, content: str) -> None:
        """添加用户消息."""
        self.messages.append(Message(role="user", content=content))
        self._trim_history()
    
    def add_assistant(self, content: str) -> None:
        """添加助手消息."""
        self.messages.append(Message(role="assistant", content=content))
        self._trim_history()
    
    def add_system(self, content: str) -> None:
        """添加系统消息."""
        # 系统消息始终保持在开头
        self.messages = [m for m in self.messages if m.role != "system"]
        self.messages.insert(0, Message(role="system", content=content))
    
    def clear(self, keep_system: bool = True) -> None:
        """清空历史."""
        if keep_system and self.messages and self.messages[0].role == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
    
    def _trim_history(self) -> None:
        """修剪历史记录."""
        if len(self.messages) > self.max_history:
            # 保留系统消息和最新的消息
            system_msg = None
            if self.messages and self.messages[0].role == "system":
                system_msg = self.messages[0]
            self.messages = self.messages[-self.max_history:]
            if system_msg and (not self.messages or self.messages[0].role != "system"):
                self.messages.insert(0, system_msg)

class LLMClient:
    """高级LLM客户端包装器.
    
    特性:
    - 自动从.env加载配置
    - 对话历史管理
    - 同步/异步流式支持
    - 工具调用支持
    - 批处理支持
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
        load_env: bool = True,
        **kwargs
    ):
        """初始化客户端.
        
        Args:
            api_key: API密钥，默认从环境变量读取
            model: 模型名称，默认从环境变量读取
            base_url: API地址，默认从环境变量读取
            provider: 提供商 (openai/anthropic)
            load_env: 是否加载.env文件
            **kwargs: 其他配置参数
        """
        if load_env:
            dotenv_loaded = load_dotenv()
        
        # 优先使用传入参数，其次环境变量
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4")
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("API key is required. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        
        # 处理base_url格式
        if self.base_url:
            self.base_url = self.base_url.rstrip("/")
            if not self.base_url.endswith("/v1"):
                self.base_url += "/v1"
        
        # 创建底层客户端
        if provider == "openai":
            self._client = CometLLM.openai(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                **kwargs
            )
        elif provider == "anthropic":
            self._client = CometLLM.anthropic(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self._session: Optional[ChatSession] = None
    
    # ========== 简单完成 ==========
    
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """简单文本完成.
        
        Args:
            prompt: 用户提示
            system: 系统提示
            **kwargs: 额外参数
        
        Returns:
            生成的文本
        """
        return self._client.complete(prompt=prompt, system=system, **kwargs)
    
    async def acomplete(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """异步简单文本完成."""
        return await self._client.acomplete(prompt=prompt, system=system, **kwargs)
    
    # ========== 聊天完成 ==========
    
    def chat(
        self,
        messages: Union[List[Message], List[Dict[str, str]], List[ChatMessage]],
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> Union[ChatResponse, Iterator[str]]:
        """聊天完成.
        
        Args:
            messages: 消息列表
            stream: 是否流式输出
            tools: 工具定义
            **kwargs: 额外参数
        
        Returns:
            完整响应或流式文本迭代器
        """
        formatted = self._format_messages(messages)
        
        if stream:
            return self._stream_response(formatted, tools, **kwargs)
        
        response = self._client.chat(formatted, stream=False, tools=tools, **kwargs)
        content = response.choices[0].message.content or "" if response.choices else ""
        return ChatResponse(
            content=content,
            usage=response.usage,
            raw_response=response
        )
    
    async def achat(
        self,
        messages: Union[List[Message], List[Dict[str, str]], List[ChatMessage]],
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> Union[ChatResponse, AsyncIterator[str]]:
        """异步聊天完成."""
        formatted = self._format_messages(messages)
        
        if stream:
            return self._astream_response(formatted, tools, **kwargs)
        
        response = await self._client.achat(formatted, stream=False, tools=tools, **kwargs)
        content = response.choices[0].message.content or "" if response.choices else ""
        return ChatResponse(
            content=content,
            usage=response.usage,
            raw_response=response
        )
    
    def _stream_response(self,messages: List[Message],tools: Optional[List[Tool]] = None,**kwargs) -> Iterator[str]:
        """生成流式响应."""
        for chunk in self._client.stream(messages, tools=tools, **kwargs):
            content = chunk.choices[0].delta.get("content", "") if chunk.choices else ""
            if content:
                yield content
    
    async def _astream_response(self,messages: List[Message],tools: Optional[List[Tool]] = None,**kwargs) -> AsyncIterator[str]:
        """生成异步流式响应."""
        async for chunk in self._client.astream(messages, tools=tools, **kwargs):
            content = chunk.choices[0].delta.get("content", "") if chunk.choices else ""
            if content:
                yield content

    def create_session(self,system: Optional[str] = None,max_history: int = 20) -> ChatSession:
        """创建新会话."""
        self._session = ChatSession(max_history=max_history)
        if system:
            self._session.add_system(system)
        return self._session
    
    def chat_with_history(self,message: str,stream: bool = False,**kwargs) -> Union[str, Iterator[str]]:
        """带历史记录的聊天.
        
        自动维护对话历史，首次调用会自动创建会话。
        """
        if self._session is None:
            self.create_session()
        
        self._session.add_user(message)
        
        if stream:
            return self._stream_with_history(**kwargs)
        
        response = self._client.chat(self._session.messages, stream=False, **kwargs)
        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""
            self._session.add_assistant(content)
        
        return content
    
    def _stream_with_history(self, **kwargs) -> Iterator[str]:
        """流式响应并更新历史."""
        full_content = []
        for chunk in self._client.stream(self._session.messages, **kwargs):
            content = chunk.choices[0].delta.get("content", "") if chunk.choices else ""
            if content:
                full_content.append(content)
                yield content
        self._session.add_assistant("".join(full_content))
    
    def clear_history(self, keep_system: bool = True) -> None:
        """清空对话历史."""
        if self._session:
            self._session.clear(keep_system)
    
    def get_history(self) -> List[ChatMessage]:
        """获取当前对话历史."""
        if not self._session:
            return []
        return [ChatMessage(role=m.role, content=str(m.content)) for m in self._session.messages]

    def batch_complete(self,prompts: List[str],system: Optional[str] = None,max_workers: int = 5,**kwargs) -> List[str]:
        """批量完成多个提示.
        
        Args:
            prompts: 提示列表
            system: 系统提示
            max_workers: 最大并发数（当前串行处理）
            **kwargs: 额外参数
        
        Returns:
            响应列表
        """
        results = []
        for prompt in prompts:
            try:
                result = self.complete(prompt, system=system, **kwargs)
                results.append(result)
            except CometLLMError as e:
                results.append(f"[Error: {e}]")
        return results
    
    async def abatch_complete(self,prompts: List[str],system: Optional[str] = None,max_concurrent: int = 5,**kwargs) -> List[str]:
        """异步批量完成.
        
        使用信号量控制并发数。
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _single(prompt: str) -> str:
            async with semaphore:
                try:
                    return await self.acomplete(prompt, system=system, **kwargs)
                except CometLLMError as e:
                    return f"[Error: {e}]"
        
        tasks = [_single(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def call_with_tools(self,prompt: str,tools: List[Tool],system: Optional[str] = None,**kwargs) -> ChatCompletion:
        """带工具调用的完成.
        
        Args:
            prompt: 用户提示
            tools: 工具定义列表
            system: 系统提示
            **kwargs: 额外参数
        
        Returns:
            原始响应对象，包含tool_calls
        """
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        
        return self._client.chat(messages, tools=tools, **kwargs)

    @staticmethod
    def _format_messages(messages: Union[List[Message], List[Dict[str, str]], List[ChatMessage]]) -> List[Message]:
        """统一格式化消息."""
        result = []
        for msg in messages:
            if isinstance(msg, Message):
                result.append(msg)
            elif isinstance(msg, ChatMessage):
                result.append(Message(role=msg.role, content=msg.content))
            elif isinstance(msg, dict):
                result.append(Message(role=msg["role"], content=msg["content"]))
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")
        return result

    def __enter__(self) -> LLMClient:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    async def __aenter__(self) -> LLMClient:
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
    
    def close(self) -> None:
        """关闭客户端."""
        self._client.close()
    
    async def aclose(self) -> None:
        """异步关闭客户端."""

def quick_complete(prompt: str,system: Optional[str] = None,**kwargs) -> str:
    """快速完成，自动从环境变量加载配置.
    
    Args:
        prompt: 用户提示
        system: 系统提示
        **kwargs: 其他参数传递给LLMClient
    
    Returns:
        生成的文本
    """
    with LLMClient(**kwargs) as client:
        return client.complete(prompt, system=system)

async def quick_acomplete(prompt: str,system: Optional[str] = None,**kwargs) -> str:
    """异步快速完成."""
    async with LLMClient(**kwargs) as client:
        return await client.acomplete(prompt, system=system)

def stream_chat(messages: List[Dict[str, str]],**kwargs) -> Iterator[str]:
    """快速流式聊天.
    
    Args:
        messages: 消息字典列表
        **kwargs: 其他参数
    
    Yields:
        文本片段
    """
    with LLMClient(**kwargs) as client:
        yield from client.chat(messages, stream=True)

def llm_function(model: Optional[str] = None,system: Optional[str] = None,temperature: float = 0.7,**client_kwargs):
    """将函数转换为LLM调用装饰器.被装饰的函数应返回prompt字符串。"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> str:
            prompt = func(*args, **kwargs)
            with LLMClient(model=model, temperature=temperature, **client_kwargs) as client:
                return client.complete(prompt, system=system)
        return wrapper
    return decorator