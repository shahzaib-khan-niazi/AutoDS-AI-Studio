"""Custom exception hierarchy for the LLM layer."""

from typing import Optional


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class LLMConfigError(LLMError):
    """Raised when LLM configuration is missing or invalid."""
    pass


class LLMProviderError(LLMError):
    """Raised when a specific model provider fails (used internally for fallback triggering)."""

    def __init__(self, message: str, model: str, status_code: Optional[int] = None, details: Optional[str] = None):
        self.model = model
        self.status_code = status_code
        super().__init__(message, details)


class LLMServiceError(LLMError):
    """Raised when both primary and fallback LLM models fail."""
    pass


class LLMValidationError(LLMError):
    """Raised when structured LLM output fails schema validation."""
    pass
