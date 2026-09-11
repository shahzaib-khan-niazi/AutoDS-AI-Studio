"""Configuration management for OpenRouter LLM integration.

Never hardcodes or prints API keys.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from core.logging import logger
from core.llm.exceptions import LLMConfigError

# Ensure environment is loaded
load_dotenv()

DEFAULT_PRIMARY_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_FALLBACK_MODEL = "minimax/minimax-m2.7:free"
DEFAULT_TIMEOUT_SECONDS = 60
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class LLMConfig:
    """Immutable LLM Configuration."""

    api_key: str
    primary_model: str = DEFAULT_PRIMARY_MODEL
    fallback_model: str = DEFAULT_FALLBACK_MODEL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    base_url: str = OPENROUTER_BASE_URL
    app_name: str = "AutoDS-AI-Studio"
    app_url: str = "https://github.com/autods-ai-studio"

    @property
    def is_configured(self) -> bool:
        """Check if an API key is present."""
        return bool(self.api_key and self.api_key.strip())

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            LLMConfigError: If OPENROUTER_API_KEY is not set.
        """
        if not self.is_configured:
            raise LLMConfigError("OPENROUTER_API_KEY is not configured.")


def load_llm_config() -> LLMConfig:
    """Load and return LLMConfig from environment variables."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    primary_model = os.getenv("OPENROUTER_PRIMARY_MODEL", DEFAULT_PRIMARY_MODEL).strip()
    fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL).strip()

    try:
        timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    except (ValueError, TypeError):
        timeout = DEFAULT_TIMEOUT_SECONDS

    cfg = LLMConfig(
        api_key=api_key,
        primary_model=primary_model if primary_model else DEFAULT_PRIMARY_MODEL,
        fallback_model=fallback_model if fallback_model else DEFAULT_FALLBACK_MODEL,
        timeout_seconds=timeout,
        base_url=OPENROUTER_BASE_URL,
    )

    if cfg.is_configured:
        logger.info(
            "LLMConfig loaded | Primary='{}' | Fallback='{}' | Timeout={}s",
            cfg.primary_model,
            cfg.fallback_model,
            cfg.timeout_seconds,
        )
    else:
        logger.warning("LLMConfig loaded without OPENROUTER_API_KEY")

    return cfg


# Global singleton
llm_config = load_llm_config()
