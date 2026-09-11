"""LLM Service — primary/fallback orchestration layer.

All application code should call LLMService methods rather than
touching the OpenRouter client directly.  The service handles:

- NVIDIA Nemotron as primary model (2 retries)
- MiniMax as fallback model (1 attempt)
- Structured JSON output with safe parsing
- Tool calling interface (LLM decides, Python executes)
- Health check
- Latency tracking
- Secure logging (never logs API keys)
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.logging import logger
from core.llm.client import OpenRouterClient
from core.llm.config import LLMConfig, llm_config
from core.llm.exceptions import (
    LLMConfigError,
    LLMProviderError,
    LLMServiceError,
    LLMValidationError,
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    """Clean result object returned by every LLMService call."""

    content: str = ""
    model: str = ""
    success: bool = False
    fallback_used: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRIMARY_MAX_RETRIES = 2
_FALLBACK_MAX_RETRIES = 1
_RETRY_DELAY_SECONDS = 1.0  # brief pause between retries

# Errors that should NOT be retried (permanent failures)
_PERMANENT_ERROR_CODES: set[int] = {400, 401, 403, 404, 422}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class LLMService:
    """Central LLM orchestration service for AutoDS AI Studio.

    Usage::

        from core.llm.service import LLMService

        svc = LLMService()
        result = svc.generate([{"role": "user", "content": "Hello"}])
        print(result.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self._config = config or llm_config
        self._client = OpenRouterClient(self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        reasoning: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> LLMResult:
        """Generate a plain-text completion with primary→fallback logic.

        Args:
            messages: Chat messages.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            reasoning: Model-specific reasoning config (e.g. ``{"enabled": True}``).
            timeout: Per-request timeout override.

        Returns:
            LLMResult with the model's answer.

        Raises:
            LLMConfigError: API key missing.
            LLMServiceError: Both primary and fallback failed.
        """
        self._config.validate()  # fail-fast on missing key
        return self._call_with_fallback(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            timeout=timeout,
        )

    def generate_structured(
        self,
        messages: list[dict[str, Any]],
        response_schema: Optional[dict[str, Any]] = None,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        reasoning: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
        retry_on_parse_failure: bool = True,
    ) -> LLMResult:
        """Generate a JSON-structured completion.

        Args:
            messages: Chat messages (should instruct JSON output).
            response_schema: Optional JSON schema hint for ``response_format``.
            temperature: Sampling temperature (default 0 for determinism).
            max_tokens: Maximum output tokens.
            reasoning: Model-specific reasoning config.
            timeout: Per-request timeout override.
            retry_on_parse_failure: Re-attempt once if JSON parsing fails.

        Returns:
            LLMResult whose ``content`` is a **validated JSON string**.

        Raises:
            LLMConfigError: API key missing.
            LLMServiceError: Both models failed.
            LLMValidationError: JSON parsing failed even after retry.
        """
        self._config.validate()

        response_format: Optional[dict[str, Any]] = {"type": "json_object"}
        if response_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": response_schema,
            }

        result = self._call_with_fallback(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            reasoning=reasoning,
            timeout=timeout,
        )

        # Validate JSON
        parsed = self._safe_parse_json(result.content)
        if parsed is not None:
            # Re-serialize to ensure clean JSON string
            result.content = json.dumps(parsed, ensure_ascii=False)
            return result

        # First parse failed — attempt one controlled retry
        if retry_on_parse_failure:
            logger.warning("JSON parse failed; retrying structured request")
            result = self._call_with_fallback(
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format=response_format,
                reasoning=reasoning,
                timeout=timeout,
            )
            parsed = self._safe_parse_json(result.content)
            if parsed is not None:
                result.content = json.dumps(parsed, ensure_ascii=False)
                return result

        raise LLMValidationError(
            message="LLM response is not valid JSON after retry.",
            details=result.content[:500],
        )

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: Optional[Any] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        reasoning: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> LLMResult:
        """Generate a completion that may include tool calls.

        The LLM decides **which** tool to call and with what arguments.
        The actual execution is handled by the caller.

        Args:
            messages: Chat messages.
            tools: OpenAI-compatible tool definitions.
            tool_choice: Tool selection strategy.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            reasoning: Model-specific reasoning config.
            timeout: Per-request timeout override.

        Returns:
            LLMResult with ``tool_calls`` populated when the model requests tools.

        Raises:
            LLMConfigError: API key missing.
            LLMServiceError: Both models failed.
        """
        self._config.validate()
        return self._call_with_fallback(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            reasoning=reasoning,
            timeout=timeout,
        )

    def health_check(self) -> LLMResult:
        """Send a tiny probe to verify connectivity.

        Returns:
            LLMResult whose content should contain ``AutoDS AI OK``.
        """
        messages = [
            {
                "role": "user",
                "content": "Reply with exactly: AutoDS AI OK",
            }
        ]
        return self.generate(messages, temperature=0.0, max_tokens=20)

    # ------------------------------------------------------------------
    # Internal retry/fallback engine
    # ------------------------------------------------------------------

    def _call_with_fallback(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        reasoning: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> LLMResult:
        """Try primary model (up to 2 attempts), then fallback (1 attempt).

        Raises:
            LLMServiceError: If all attempts fail.
        """
        primary = self._config.primary_model
        fallback = self._config.fallback_model

        # --- Primary attempts ---
        last_error: Optional[Exception] = None
        for attempt in range(1, _PRIMARY_MAX_RETRIES + 1):
            logger.info(
                "LLM request started | model={} | attempt={}/{}",
                primary,
                attempt,
                _PRIMARY_MAX_RETRIES,
            )
            start = time.perf_counter()
            try:
                raw = self._client.complete(
                    model=primary,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice,
                    reasoning=reasoning,
                    timeout=timeout,
                )
                latency = (time.perf_counter() - start) * 1000
                result = self._build_result(raw, fallback_used=False, latency_ms=latency)
                self._log_success(result)
                return result
            except LLMProviderError as exc:
                latency = (time.perf_counter() - start) * 1000
                last_error = exc
                if self._is_permanent_error(exc):
                    logger.error(
                        "Permanent error on primary model (status={}); skipping retries",
                        exc.status_code,
                    )
                    break  # go straight to fallback
                logger.warning(
                    "Primary model attempt {}/{} failed ({:.0f}ms): {}",
                    attempt,
                    _PRIMARY_MAX_RETRIES,
                    latency,
                    exc.message,
                )
                if attempt < _PRIMARY_MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_SECONDS)

        # --- Fallback attempt ---
        logger.warning(
            "Primary model failed; switching to fallback model '{}'",
            fallback,
        )

        for attempt in range(1, _FALLBACK_MAX_RETRIES + 1):
            logger.info(
                "LLM request started | model={} | fallback attempt={}/{}",
                fallback,
                attempt,
                _FALLBACK_MAX_RETRIES,
            )
            start = time.perf_counter()
            try:
                raw = self._client.complete(
                    model=fallback,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice,
                    reasoning=reasoning,
                    timeout=timeout,
                )
                latency = (time.perf_counter() - start) * 1000
                result = self._build_result(raw, fallback_used=True, latency_ms=latency)
                self._log_success(result)
                return result
            except LLMProviderError as exc:
                latency = (time.perf_counter() - start) * 1000
                last_error = exc
                logger.error(
                    "Fallback model attempt {}/{} failed ({:.0f}ms): {}",
                    attempt,
                    _FALLBACK_MAX_RETRIES,
                    latency,
                    exc.message,
                )

        # --- Both failed ---
        raise LLMServiceError(
            message="All LLM models failed. Please try again later.",
            details=str(last_error) if last_error else "Unknown error",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        raw: dict[str, Any],
        *,
        fallback_used: bool,
        latency_ms: float,
    ) -> LLMResult:
        """Convert raw client response dict into an LLMResult."""
        return LLMResult(
            content=raw.get("content", ""),
            model=raw.get("model_used", ""),
            success=True,
            fallback_used=fallback_used,
            usage=raw.get("usage", {}),
            finish_reason=raw.get("finish_reason", ""),
            tool_calls=raw.get("tool_calls", []),
            latency_ms=round(latency_ms, 1),
        )

    @staticmethod
    def _is_permanent_error(exc: LLMProviderError) -> bool:
        """Return True if the error should NOT be retried."""
        if exc.status_code and exc.status_code in _PERMANENT_ERROR_CODES:
            return True
        return False

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[Any]:
        """Try to parse text as JSON. Return None on failure."""
        if not text:
            return None
        text_str = text.strip()
        try:
            return json.loads(text_str)
        except (json.JSONDecodeError, TypeError):
            pass

        # Search for ``` or ```json code blocks anywhere in text
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_str, re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                pass

        # Search for first { and last } or first [ and last ]
        first_brace = text_str.find("{")
        last_brace = text_str.rfind("}")
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            candidate = text_str[first_brace:last_brace + 1]
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    @staticmethod
    def _log_success(result: LLMResult) -> None:
        """Log a successful response (no secrets)."""
        usage_str = ""
        if result.usage:
            usage_str = (
                f" | tokens={result.usage.get('total_tokens', '?')}"
            )
        logger.info(
            "LLM request successful | model={} | fallback={} | {:.0f}ms{}",
            result.model,
            result.fallback_used,
            result.latency_ms,
            usage_str,
        )
