![Logo](https://avatars.githubusercontent.com/u/184814437?v=4&size=64 "CometLLM-Client")
<div align='center'> 
    <img src="https://avatars.githubusercontent.com/u/184814437?v=4&size=60">
    <font size="70"> Comet LLM Client V0.0.5-Alpha </font>
</div>

高性能LLM客户端，聚合OpenAI和Anthropic API，提供统一接口和优化性能，使得开发者无需单独适配服务商，尽可能提高效率

## 特性

- 🚀 高性能异步支持
- 🔧 统一API接口（兼容OpenAI和Anthropic）
- 💾 智能HTTP连接池
- 🔄 自动重试和错误处理
- 📊 完整的类型提示
- 🔒 Pydantic数据验证
- 🛠️ 工具调用支持
- 📡 流式响应支持
- ✨ 优化的接口，无需手动适配每个服务商

## 安装

```bash
pip install -e .
```

## 快速开始

### 使用OpenAI

```python
from CometLLM import CometLLM

# 方法1: 使用工厂方法
client = CometLLM.openai(api_key="your-api-key", model="gpt-4")

# 方法2: 使用配置字典
client = CometLLM({
    "provider": "openai",
    "api_key": "your-api-key",
    "model": "gpt-4",
    "temperature": 0.7,
})

# 简单完成
response = client.complete("Hello, how are you?")
print(response)

# 聊天完成
from CometLLM import Message

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is Python?"},
]

response = client.chat(messages)
print(response.choices[0].message.content)
```

### 使用Anthropic (Claude)

```python
from CometLLM import CometLLM

client = CometLLM.anthropic(api_key="your-api-key", model="claude-3-opus-20240229")

response = client.complete("Hello, how are you?")
print(response)
```

### 流式响应

```python
# 同步流式
for chunk in client.stream([{"role": "user", "content": "Tell me a story"}]):
    content = chunk.choices[0].delta.get("content", "")
    print(content, end="", flush=True)

# 异步流式
async for chunk in client.astream([{"role": "user", "content": "Tell me a story"}]):
    content = chunk.choices[0].delta.get("content", "")
    print(content, end="", flush=True)
```

### 异步使用

```python
import asyncio
from CometLLM import CometLLM

async def main():
    client = CometLLM.openai(api_key="your-api-key")
    
    # 异步完成
    response = await client.acomplete("Hello!")
    print(response)
    
    # 异步聊天
    response = await client.achat([
        {"role": "user", "content": "What is AI?"}
    ])
    print(response.choices[0].message.content)
    
    await client.aclose()

asyncio.run(main())
```

### 工具调用

```python
from CometLLM import CometLLM, Tool, ToolFunction

client = CometLLM.openai(api_key="your-api-key", model="gpt-4")

tools = [
    Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        )
    )
]

response = client.chat(
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
)

# 检查工具调用
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function: {tool_call['function']['name']}")
        print(f"Arguments: {tool_call['function']['arguments']}")
```

## 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `provider` | str | 必填 | 提供商: `openai` 或 `anthropic` |
| `api_key` | str | 必填 | API密钥 |
| `model` | str | 必填 | 模型名称 |
| `base_url` | str | None | 自定义API地址 |
| `temperature` | float | 0.7 | 采样温度 (0-2) |
| `max_tokens` | int | None | 最大生成token数 |
| `top_p` | float | 1.0 | 核采样参数 |
| `timeout` | float | 60.0 | 请求超时时间(秒) |
| `max_retries` | int | 3 | 最大重试次数 |
| `retry_delay` | float | 1.0 | 重试延迟(秒) |

## 错误处理

```python
from CometLLM import CometLLM
from CometLLM.exceptions import AuthenticationError, RateLimitError, APIError

client = CometLLM.openai(api_key="your-api-key")

try:
    response = client.complete("Hello")
except AuthenticationError as e:
    print(f"认证失败: {e.message}")
except RateLimitError as e:
    print(f"速率限制: {e.message}")
except APIError as e:
    print(f"API错误: {e.message} (状态码: {e.status_code})")
```

## 项目结构

```
CometLLM/
├── __init__.py          # 包入口
├── client.py            # 统一客户端
├── types.py             # 类型定义
├── exceptions.py        # 异常定义
├── base.py              # 基础客户端
└── providers/
    ├── __init__.py
    ├── openai.py        # OpenAI实现
    └── anthropic.py     # Anthropic实现
```

## 许可证

MIT
