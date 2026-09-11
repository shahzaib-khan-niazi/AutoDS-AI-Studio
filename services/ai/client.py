"""AI Client for AutoDS AI Studio.

Wrapper for LLM calls with error handling, timeouts, and structured JSON parsing.
Delegates all LLM calls to core.llm.service.LLMService (OpenRouter integration).
"""

import json
from typing import Any

from core.exceptions import AIServiceError
from core.llm.config import llm_config
from core.llm.service import LLMService
from core.llm.exceptions import LLMError
from core.logging import logger


class AIClient:
    """Client for executing structured LLM requests via LLMService."""

    @classmethod
    def call_structured(
        cls,
        prompt: str,
        system_instruction: str = "You are a professional Data Science AI Assistant. Return only valid JSON.",
    ) -> dict[str, Any]:
        """Send a prompt to the LLM and parse the structured JSON response.

        Args:
            prompt: User prompt.
            system_instruction: System prompt.

        Returns:
            Parsed JSON dictionary.

        Raises:
            AIServiceError: If API key is missing, network error, or invalid JSON.
        """
        if not llm_config.is_configured:
            raise AIServiceError(
                message="AI service is not configured.",
                details="Please set OPENROUTER_API_KEY in your .env file.",
            )

        logger.info("Sending structured AI request via LLMService")

        try:
            svc = LLMService()
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ]
            result = svc.generate_structured(messages)
            parsed = json.loads(result.content)
            return parsed

        except LLMError as e:
            logger.error("AI service error: {}", e.message)
            raise AIServiceError(
                message=f"AI service call failed: {e.message}",
                details=e.details or "",
            ) from e
        except json.JSONDecodeError as e:
            logger.error("Failed to parse AI JSON response: {}", str(e))
            raise AIServiceError(
                message="AI returned an invalid JSON response.",
                details=str(e),
            ) from e
        except Exception as e:
            logger.exception("AI request failed: {}", str(e))
            raise AIServiceError(
                message="AI service request failed.",
                details=str(e),
            ) from e
