"""基础HTTP客户端和通用功能."""
from __future__ import annotations
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, Optional, TypeVar, Union
import httpx
from tenacity import retry,retry_if_exception_type,stop_after_attempt,wait_exponential
from .exceptions import APIError,AuthenticationError,CometLLMError,RateLimitError,TimeoutError
from .types import ChatCompletion,ChatCompletionChunk,LLMConfig,Message,Tool
T = TypeVar("T")

class BaseClient(ABC):
    """基础HTTP客户端."""

    DEFAULT_BASE_URLS = {"openai": "https://api.openai.com/v1","anthropic": "https://api.anthropic.com/v1",}

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or self.DEFAULT_BASE_URLS[config.provider.value]
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头."""
        return {"Content-Type": "application/json","Authorization": f"Bearer {self.config.api_key}",}

    def _get_sync_client(self) -> httpx.Client:
        """获取同步HTTP客户端."""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                timeout=self.config.timeout,
                headers=self._get_headers(),)
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        """获取异步HTTP客户端."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(base_url=self.base_url,
                timeout=self.config.timeout,
                headers=self._get_headers(),)
        return self._async_client

    def _handle_error(self, response: httpx.Response) -> None:
        """处理HTTP错误."""
        status_code = response.status_code
        try:
            body = response.json()
        except Exception:
            body = {"error": {"message": response.text}}

        if isinstance(body, dict):
            error_obj = body.get("error", {})
            if isinstance(error_obj, str):
                message = error_obj
            elif isinstance(error_obj, dict):
                message = error_obj.get("message", response.text)
            else:
                message = str(error_obj)
        else:
            message = str(body)

        if status_code == 401:
            raise AuthenticationError(message, status_code, body)
        elif status_code == 429:
            raise RateLimitError(message, status_code, body)
        elif status_code >= 500:
            raise APIError(message, status_code, body)
        elif status_code >= 400:
            raise APIError(message, status_code, body)

    def _retry_config(self):
        """重试配置."""
        return retry(stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=self.config.retry_delay, min=1, max=10),
            retry=retry_if_exception_type((APIError, RateLimitError, TimeoutError)),
            reraise=True,)

    def close(self) -> None:
        """关闭同步客户端."""
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()

    async def aclose(self) -> None:
        """关闭异步客户端."""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    @abstractmethod
    def chat_completion(self,messages: list[Message],stream: bool = False,tools: Optional[list[Tool]] = None,**kwargs) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """同步聊天完成."""
        pass

    @abstractmethod
    async def achat_completion(self,messages: list[Message],stream: bool = False,tools: Optional[list[Tool]] = None,**kwargs) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """异步聊天完成."""
        pass