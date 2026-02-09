"""OpenAI API 客户端实现."""
from __future__ import annotations
import json
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
import httpx
from ..base import BaseClient
from ..exceptions import APIError, StreamError
from ..types import ChatCompletion,ChatCompletionChunk,Choice,ChoiceDelta,LLMConfig,Message,Role,Tool,Usage

class OpenAIClient(BaseClient):
    """OpenAI API 客户端."""

    def _get_headers(self) -> Dict[str, str]:
        """获取OpenAI请求头."""
        return {"Content-Type": "application/json","Authorization": f"Bearer {self.config.api_key}",}

    def _build_payload(self,messages: List[Message],stream: bool = False,tools: Optional[List[Tool]] = None,**kwargs) -> Dict[str, Any]:
        """构建请求体."""
        payload = {"model": self.config.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),"stream": stream,}
        
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in tools]
            if self.config.tool_choice:
                payload["tool_choice"] = self.config.tool_choice
        
        for key in ["stop", "presence_penalty", "frequency_penalty", "logit_bias", "user"]:
            if key in kwargs:
                payload[key] = kwargs[key]
        return payload
    
    def _parse_response(self, data: Dict[str, Any]) -> ChatCompletion:
        """解析响应."""
        choices = []
        for c in data.get("choices", []):
            msg_data = c.get("message", {})
            message = Message(role=Role(msg_data.get("role", "assistant")),
                content=msg_data.get("content", ""),
                tool_calls=msg_data.get("tool_calls"),)
            choices.append(Choice(index=c.get("index", 0),
                message=message,
                finish_reason=c.get("finish_reason"),))
        
        usage = None
        if "usage" in data:
            usage = Usage(**data["usage"])
        
        return ChatCompletion(id=data.get("id", ""),
            created=data.get("created", int(time.time())),
            model=data.get("model", self.config.model),
            choices=choices,usage=usage,)
    
    def _parse_stream_chunk(self, line: str) -> Optional[ChatCompletionChunk]:
        """解析流式响应块."""
        if not line.startswith("data: "):
            return None
        
        data_str = line[6:]
        if data_str == "[DONE]":
            return None
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return None
        
        choices = []
        for c in data.get("choices", []):
            delta = c.get("delta", {})
            choices.append(ChoiceDelta(
                index=c.get("index", 0),
                delta=delta,
                finish_reason=c.get("finish_reason"),))
        
        usage = None
        if "usage" in data:
            usage = Usage(**data["usage"])
        
        return ChatCompletionChunk(id=data.get("id", ""),
            created=data.get("created", int(time.time())),
            model=data.get("model", self.config.model),
            choices=choices,usage=usage,)
    
    def chat_completion(self,messages: List[Message],stream: bool = False,tools: Optional[List[Tool]] = None,**kwargs) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """同步聊天完成."""
        payload = self._build_payload(messages, stream, tools, **kwargs)
        client = self._get_sync_client()
        if stream:
            return self._stream_chat_completion(client, payload)
        
        response = client.post("/chat/completions", json=payload)
        if response.status_code != 200:
            self._handle_error(response)
        return self._parse_response(response.json())
    
    def _stream_chat_completion(self,client: httpx.Client,payload: Dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        """流式聊天完成."""
        try:
            with client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    self._handle_error(response)
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    chunk = self._parse_stream_chunk(line)
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as e:
            raise APIError(f"Request timeout: {e}")
        except Exception as e:
            raise StreamError(f"Stream error: {e}")
    
    async def achat_completion(self,messages: List[Message],stream: bool = False,tools: Optional[List[Tool]] = None,**kwargs) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """异步聊天完成."""
        payload = self._build_payload(messages, stream, tools, **kwargs)
        client = self._get_async_client()
        if stream:
            return self._astream_chat_completion(client, payload)
        
        response = await client.post("/chat/completions", json=payload)
        if response.status_code != 200:
            self._handle_error(response)
        
        return self._parse_response(response.json())

    async def _astream_chat_completion(self,client: httpx.AsyncClient,payload: Dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        """异步流式聊天完成."""
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._handle_error(response)
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    chunk = self._parse_stream_chunk(line)
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as e:
            raise APIError(f"Request timeout: {e}")
        except Exception as e:
            raise StreamError(f"Stream error: {e}")