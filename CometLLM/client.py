"""统一LLM客户端."""
from __future__ import annotations
from typing import Any, AsyncIterator, Iterator, List, Optional, Union
from .base import BaseClient
from .exceptions import ValidationError
from .providers import AnthropicClient, OpenAIClient
from .types import ChatCompletion,ChatCompletionChunk,LLMConfig,Message,Provider,Tool

class CometLLM:
    """统一LLM客户端，支持OpenAI和Anthropic API."""
    
    def __init__(self, config: Union[LLMConfig, dict]):
        """初始化客户端.
        
        Args:
            config: LLM配置对象或字典
        """
        if isinstance(config, dict):
            config = LLMConfig(**config)
        self.config = config
        self._client: Optional[BaseClient] = None
        self._init_client()
    
    def _init_client(self) -> None:
        """初始化底层客户端."""
        if self.config.provider == Provider.OPENAI:
            self._client = OpenAIClient(self.config)
        elif self.config.provider == Provider.ANTHROPIC:
            self._client = AnthropicClient(self.config)
        else:
            raise ValidationError(f"不支持的提供商: {self.config.provider}")
    
    @classmethod
    def openai(cls,api_key: str,model: str = "gpt-5-3",base_url: Optional[str] = None,**kwargs) -> "CometLLM":
        """快速创建OpenAI客户端.
        
        Args:
            api_key: API密钥
            model: 模型名称
            base_url: 自定义API地址
            **kwargs: 其他配置参数
        """
        config = LLMConfig(provider=Provider.OPENAI,
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs)
        return cls(config)
    
    @classmethod
    def anthropic(cls,api_key: str,model: str = "claude-4-6-opus",base_url: Optional[str] = None,**kwargs) -> "CometLLM":
        """快速创建Anthropic客户端.
        
        Args:
            api_key: API密钥
            model: 模型名称
            base_url: 自定义API地址
            **kwargs: 其他配置参数
        """
        config = LLMConfig(provider=Provider.ANTHROPIC,
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs)
        return cls(config)
    
    def chat(self,messages: Union[List[Message], List[dict]],stream: bool = False,tools: Optional[List[Tool]] = None,**kwargs) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """同步聊天完成.
        
        Args:
            messages: 消息列表
            stream: 是否流式响应
            tools: 工具定义列表
            **kwargs: 额外参数覆盖配置
        
        Returns:
            聊天完成响应或流式迭代器
        """
        if not self._client:
            raise RuntimeError("客户端未初始化")
        
        # 转换字典消息为Message对象
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(Message(**msg))
            else:
                formatted_messages.append(msg)
        
        return self._client.chat_completion(
            messages=formatted_messages,
            stream=stream,
            tools=tools,**kwargs)
    
    async def achat(self,messages: Union[List[Message], List[dict]],stream: bool = False,tools: Optional[List[Tool]] = None,**kwargs) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """异步聊天完成.
        
        Args:
            messages: 消息列表
            stream: 是否流式响应
            tools: 工具定义列表
            **kwargs: 额外参数覆盖配置
        
        Returns:
            聊天完成响应或异步流式迭代器
        """
        if not self._client:
            raise RuntimeError("客户端未初始化")
        
        # 转换字典消息为Message对象
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(Message(**msg))
            else:
                formatted_messages.append(msg)
        
        return await self._client.achat_completion(
            messages=formatted_messages,
            stream=stream,
            tools=tools,
            **kwargs)
    
    def complete(self,prompt: str,system: Optional[str] = None,**kwargs) -> str:
        """简单完成接口.
        
        Args:
            prompt: 用户提示
            system: 系统提示
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        response = self.chat(messages, stream=False, **kwargs)
        if isinstance(response, ChatCompletion):
            return response.choices[0].message.content or ""
        return ""
    
    async def acomplete(self,prompt: str,system: Optional[str] = None,**kwargs) -> str:
        """异步简单完成接口.
        
        Args:
            prompt: 用户提示
            system: 系统提示
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        
        response = await self.achat(messages, stream=False, **kwargs)
        if isinstance(response, ChatCompletion):
            return response.choices[0].message.content or ""
        return ""
    
    def stream(self,messages: Union[List[Message], List[dict]],**kwargs) -> Iterator[ChatCompletionChunk]:
        """流式聊天接口.
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
        
        Yields:
            响应块
        """
        result = self.chat(messages, stream=True, **kwargs)
        if isinstance(result, Iterator):
            yield from result
    
    async def astream(self,messages: Union[List[Message], List[dict]],**kwargs) -> AsyncIterator[ChatCompletionChunk]:
        """异步流式聊天接口.
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
        
        Yields:
            响应块
        """
        result = await self.achat(messages, stream=True, **kwargs)
        if isinstance(result, AsyncIterator):
            async for chunk in result:
                yield chunk
    
    def close(self) -> None:
        """关闭客户端."""
        if self._client:
            self._client.close()
    
    async def aclose(self) -> None:
        """异步关闭客户端."""
        if self._client:
            await self._client.aclose()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()