"""异常定义."""

class CometLLMError(Exception):
    """基础异常."""

    def __init__(self, message: str, status_code: int = None, response_body: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body

class AuthenticationError(CometLLMError):
    """认证错误 (401)."""
    pass

class RateLimitError(CometLLMError):
    """速率限制错误 (429)."""
    pass

class APIError(CometLLMError):
    """API错误."""
    pass

class TimeoutError(CometLLMError):
    """超时错误."""
    pass

class ValidationError(CometLLMError):
    """参数验证错误."""
    pass

class StreamError(CometLLMError):
    """流式响应错误."""
    pass