"""Anthropic API 客户端实现."""
from __future__ import annotations
import json
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
import httpx
from ..base import BaseClient
from ..exceptions import APIError, StreamError
from ..types import ChatCompletion,ChatCompletionChunk,Choice,ChoiceDelta,LLMConfig,Message,Role,Tool,Usage

class AnthropicClient(BaseClient):
    """Anthropic API 客户端."""
    
    def _get_headers(self) -> Dict[str, str]:
        """获取Anthropic请求头."""
        return {"Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",}
    
    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """转换消息格式为Anthropic格式.
        
        Returns:
            (system_prompt, anthropic_messages)
        """
        system_prompt = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content if isinstance(msg.content, str) else None
                continue
            
            content = msg.content
            if isinstance(content, str):
                anthropic_content = content
            else:
                anthropic_content = content
            
            anthropic_msg = {"role": "user" if msg.role == Role.USER else "assistant","content": anthropic_content,}
            if msg.tool_calls:
                anthropic_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                anthropic_msg["tool_call_id"] = msg.tool_call_id
            
            anthropic_messages.append(anthropic_msg)
        return system_prompt, anthropic_messages
    
    def _build_payload(self,messages: List[Message],stream: bool = False,tools: Optional[List[Tool]] = None,**kwargs) -> Dict[str, Any]:
        """构建请求体."""
        system_prompt, anthropic_messages = self._convert_messages(messages)
        
        payload: Dict[str, Any] = {"model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens) or 4096,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": stream,}
        
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = [t.model_dump(exclude_none=True)["function"] for t in tools]
            payload["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in payload["tools"]]
        
        for key in ["stop_sequences", "top_k", "metadata"]:
            if key in kwargs:
                payload[key] = kwargs[key]
        
        return payload
    
    def _parse_response(self, data: Dict[str, Any]) -> ChatCompletion:
        """解析Anthropic响应为统一格式."""
        content = ""
        tool_calls = None
        
        if data.get("content"):
            for block in data["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        }
                    })
        
        message = Message(role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,)
        
        usage = None
        if "usage" in data:
            usage_data = data["usage"]
            usage = Usage(prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),)
        
        return ChatCompletion(id=data.get("id", ""),
            created=int(time.time()),
            model=data.get("model", self.config.model),
            choices=[Choice(message=message, finish_reason=data.get("stop_reason"))],
            usage=usage,)
    
    def _parse_stream_chunk(self, event: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        """解析流式事件为统一格式."""
        event_type = event.get("type")
        
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type")
            
            if delta_type == "text_delta":
                return ChatCompletionChunk(id=event.get("message", {}).get("id", ""),
                    created=int(time.time()),
                    model=self.config.model,
                    choices=[ChoiceDelta(delta={"content": delta.get("text", "")})],)
            elif delta_type == "input_json_delta":
                return ChatCompletionChunk(id=event.get("message", {}).get("id", ""),
                    created=int(time.time()),
                    model=self.config.model,
                    choices=[ChoiceDelta(delta={"tool_input": delta.get("partial_json", "")})],)
        
        elif event_type == "message_stop":
            return ChatCompletionChunk(id="",
                created=int(time.time()),
                model=self.config.model,
                choices=[ChoiceDelta(delta={}, finish_reason="stop")],)
        
        elif event_type == "message_start":
            msg = event.get("message", {})
            return ChatCompletionChunk(id=msg.get("id", ""),
                created=int(time.time()),
                model=msg.get("model", self.config.model),
                choices=[ChoiceDelta(delta={"role": "assistant"})],)
        return None
    
    def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """同步聊天完成."""
        payload = self._build_payload(messages, stream, tools, **kwargs)
        client = self._get_sync_client()
        
        if stream:
            return self._stream_chat_completion(client, payload)
        
        response = client.post("/messages", json=payload)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        return self._parse_response(response.json())
    
    def _stream_chat_completion(
        self,
        client: httpx.Client,
        payload: Dict[str, Any]
    ) -> Iterator[ChatCompletionChunk]:
        """流式聊天完成."""
        try:
            with client.stream("POST", "/messages", json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    self._handle_error(response)
                
                buffer = ""
                for chunk in response.iter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                                parsed = self._parse_stream_chunk(event)
                                if parsed:
                                    yield parsed
                            except json.JSONDecodeError:
                                continue
                        elif line.startswith("event: "):
                            continue
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
        
        response = await client.post("/messages", json=payload)
        if response.status_code != 200:
            self._handle_error(response)
        
        return self._parse_response(response.json())
    
    async def _astream_chat_completion(self,client: httpx.AsyncClient,payload: Dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        """异步流式聊天完成."""
        try:
            async with client.stream("POST", "/messages", json=payload) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._handle_error(response)
                
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                                parsed = self._parse_stream_chunk(event)
                                if parsed:
                                    yield parsed
                            except json.JSONDecodeError:
                                continue
                        elif line.startswith("event: "):
                            continue
        except httpx.TimeoutException as e:
            raise APIError(f"Request timeout: {e}")
        except Exception as e:
            raise StreamError(f"Stream error: {e}")