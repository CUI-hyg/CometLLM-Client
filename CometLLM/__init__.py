"""CometLLM - 高性能LLM客户端，兼容OpenAI和Anthropic API."""
from .client import CometLLM
from .types import Message,ChatCompletion,ChatCompletionChunk,Usage,LLMConfig,Provider
from .exceptions import CometLLMError,AuthenticationError,RateLimitError,APIError,TimeoutError
from .wrapper import LLMClient,ChatMessage,ChatResponse,ChatSession,quick_complete,quick_acomplete,stream_chat,llm_function

__version__ = "0.0.5"
__all__ = ["CometLLM","Message","ChatCompletion","ChatCompletionChunk","Usage","LLMConfig","Provider","CometLLMError","AuthenticationError","RateLimitError","APIError","TimeoutError","LLMClient","ChatMessage","ChatResponse","ChatSession","quick_complete","quick_acomplete","stream_chat","llm_function"]