"""Application configuration for AutoDS AI Studio.

Reads settings from environment variables via .env file.
Never hardcodes API keys or model names.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from core.logging import logger

# Load .env file if present
load_dotenv()


@dataclass(frozen=True)
class AIConfig:
    """Configuration for AI/LLM services."""
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 60

    @property
    def is_configured(self) -> bool:
        """Check if AI service has a valid API key."""
        return len(self.api_key) > 0


@dataclass(frozen=True)
class UploadConfig:
    """Configuration for file upload limits."""
    max_file_size_mb: int = 200
    supported_extensions: list[str] = field(
        default_factory=lambda: [".csv", ".xlsx", ".xls", ".parquet"]
    )


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration."""
    ai: AIConfig = field(default_factory=AIConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    debug: bool = False


def load_config() -> AppConfig:
    """Load application configuration from environment variables.

    Returns:
        AppConfig with values populated from environment.
    """
    ai_config = AIConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
        timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
    )

    upload_config = UploadConfig(
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "200")),
    )

    debug = os.getenv("DEBUG", "false").lower() == "true"

    config = AppConfig(ai=ai_config, upload=upload_config, debug=debug)

    if config.ai.is_configured:
        logger.info("AI service configured with model: {}", config.ai.model)
    else:
        logger.info("AI service not configured (no API key)")

    return config


# Singleton config instance
config = load_config()
