"""Unit tests for core.llm (config, client, service).

All tests use mocks — no real API calls are made.
Run with: pytest tests/test_llm.py -v
"""

import json
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from core.llm.config import LLMConfig, load_llm_config, DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL
from core.llm.client import OpenRouterClient
from core.llm.service import LLMService, LLMResult
from core.llm.exceptions import (
    LLMConfigError,
    LLMProviderError,
    LLMServiceError,
    LLMValidationError,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Configuration tests
# ═══════════════════════════════════════════════════════════════════════


class TestLLMConfig:
    """Tests for core.llm.config.LLMConfig."""

    def test_config_loads_correctly(self) -> None:
        """Configuration should be created with given values."""
        cfg = LLMConfig(
            api_key="test-key-123",
            primary_model="nvidia/nemotron-3-super-120b-a12b:free",
            fallback_model="minimax/minimax-m2.7:free",
            timeout_seconds=30,
        )
        assert cfg.api_key == "test-key-123"
        assert cfg.primary_model == "nvidia/nemotron-3-super-120b-a12b:free"
        assert cfg.fallback_model == "minimax/minimax-m2.7:free"
        assert cfg.timeout_seconds == 30
        assert cfg.is_configured is True

    def test_missing_api_key_raises_config_error(self) -> None:
        """Validate should raise LLMConfigError when API key is empty."""
        cfg = LLMConfig(api_key="")
        with pytest.raises(LLMConfigError, match="OPENROUTER_API_KEY is not configured"):
            cfg.validate()

    def test_whitespace_api_key_is_not_configured(self) -> None:
        """Whitespace-only key should not count as configured."""
        cfg = LLMConfig(api_key="   ")
        assert cfg.is_configured is False

    def test_model_names_load_with_defaults(self) -> None:
        """Default model names should match expected values."""
        cfg = LLMConfig(api_key="k")
        assert cfg.primary_model == DEFAULT_PRIMARY_MODEL
        assert cfg.fallback_model == DEFAULT_FALLBACK_MODEL

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "env-key-xyz",
        "OPENROUTER_PRIMARY_MODEL": "custom/primary:free",
        "OPENROUTER_FALLBACK_MODEL": "custom/fallback:free",
        "LLM_TIMEOUT_SECONDS": "90",
    })
    def test_load_llm_config_from_env(self) -> None:
        """load_llm_config should read from environment variables."""
        cfg = load_llm_config()
        assert cfg.api_key == "env-key-xyz"
        assert cfg.primary_model == "custom/primary:free"
        assert cfg.fallback_model == "custom/fallback:free"
        assert cfg.timeout_seconds == 90


# ═══════════════════════════════════════════════════════════════════════
# 2. Client tests
# ═══════════════════════════════════════════════════════════════════════


class TestOpenRouterClient:
    """Tests for core.llm.client.OpenRouterClient."""

    def _make_config(self) -> LLMConfig:
        return LLMConfig(api_key="test-key")

    def test_client_initialization(self) -> None:
        """Client should initialize without errors."""
        client = OpenRouterClient(self._make_config())
        assert client._client is None  # lazy init

    def test_client_validates_config_on_get(self) -> None:
        """get_client should raise if API key is missing."""
        bad_cfg = LLMConfig(api_key="")
        client = OpenRouterClient(bad_cfg)
        with pytest.raises(LLMConfigError):
            client.get_client()

    def test_sanitize_content_removes_think_tags(self) -> None:
        """Private reasoning tags should be stripped."""
        raw = "Hello <think>internal reasoning</think> World"
        clean = OpenRouterClient._sanitize_content(raw)
        assert "<think>" not in clean
        assert "World" in clean
        assert "Hello" in clean

    def test_sanitize_content_removes_thought_tags(self) -> None:
        """<thought> variant should also be stripped."""
        raw = "<thought>step 1\nstep 2</thought>Final answer."
        clean = OpenRouterClient._sanitize_content(raw)
        assert "<thought>" not in clean
        assert "Final answer." in clean


# ═══════════════════════════════════════════════════════════════════════
# 3. Service tests
# ═══════════════════════════════════════════════════════════════════════


def _mock_raw_response(content: str = "Hello", model: str = "test-model") -> dict:
    """Build a fake raw response dict like OpenRouterClient.complete returns."""
    return {
        "content": content,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "tool_calls": [],
        "model_used": model,
    }


class TestLLMService:
    """Tests for core.llm.service.LLMService."""

    def _make_service(self) -> LLMService:
        cfg = LLMConfig(
            api_key="test-key",
            primary_model="primary/model",
            fallback_model="fallback/model",
        )
        return LLMService(config=cfg)

    # --- Test 4: Primary model is selected first ---
    @patch.object(OpenRouterClient, "complete")
    def test_primary_model_selected_first(self, mock_complete: MagicMock) -> None:
        """generate() should call the primary model first."""
        mock_complete.return_value = _mock_raw_response(model="primary/model")
        svc = self._make_service()

        result = svc.generate([{"role": "user", "content": "hi"}])
        assert result.success is True
        assert result.fallback_used is False
        # First call should use primary model
        call_args = mock_complete.call_args
        assert call_args[1]["model"] == "primary/model" or call_args[0][0] == "primary/model"

    # --- Test 5: Fallback is triggered after primary failure ---
    @patch.object(OpenRouterClient, "complete")
    def test_fallback_triggered_on_primary_failure(self, mock_complete: MagicMock) -> None:
        """If primary fails, fallback model should be tried."""
        mock_complete.side_effect = [
            LLMProviderError("fail 1", model="primary/model", status_code=503),
            LLMProviderError("fail 2", model="primary/model", status_code=503),
            _mock_raw_response(model="fallback/model"),
        ]
        svc = self._make_service()
        result = svc.generate([{"role": "user", "content": "hi"}])

        assert result.success is True
        assert result.fallback_used is True
        assert result.model == "fallback/model"

    # --- Test 6: Successful primary does NOT call fallback ---
    @patch.object(OpenRouterClient, "complete")
    def test_successful_primary_no_fallback(self, mock_complete: MagicMock) -> None:
        """On primary success, complete() should only be called once."""
        mock_complete.return_value = _mock_raw_response(model="primary/model")
        svc = self._make_service()
        svc.generate([{"role": "user", "content": "hi"}])

        assert mock_complete.call_count == 1

    # --- Test 7: Both failures produce clean error ---
    @patch.object(OpenRouterClient, "complete")
    def test_both_failures_raise_service_error(self, mock_complete: MagicMock) -> None:
        """When both primary and fallback fail, LLMServiceError is raised."""
        mock_complete.side_effect = LLMProviderError(
            "always fails", model="any", status_code=503,
        )
        svc = self._make_service()

        with pytest.raises(LLMServiceError, match="All LLM models failed"):
            svc.generate([{"role": "user", "content": "hi"}])

    # --- Test 8: API key never in logs/errors ---
    @patch.object(OpenRouterClient, "complete")
    def test_api_key_never_in_error_messages(self, mock_complete: MagicMock) -> None:
        """Error messages must not contain the API key."""
        api_key = "sk-secret-key-12345"
        cfg = LLMConfig(api_key=api_key, primary_model="p", fallback_model="f")
        svc = LLMService(config=cfg)

        mock_complete.side_effect = LLMProviderError(
            "provider down", model="p", status_code=500,
        )
        with pytest.raises(LLMServiceError) as exc_info:
            svc.generate([{"role": "user", "content": "hi"}])

        assert api_key not in str(exc_info.value)
        assert api_key not in str(exc_info.value.message)
        if exc_info.value.details:
            assert api_key not in exc_info.value.details

    # --- Test 9: Structured JSON parsed correctly ---
    @patch.object(OpenRouterClient, "complete")
    def test_structured_json_parsed(self, mock_complete: MagicMock) -> None:
        """generate_structured should parse valid JSON."""
        valid_json = json.dumps({"columns": ["a", "b"], "count": 5})
        mock_complete.return_value = _mock_raw_response(content=valid_json)
        svc = self._make_service()

        result = svc.generate_structured(
            [{"role": "user", "content": "give json"}],
        )
        assert result.success is True
        parsed = json.loads(result.content)
        assert parsed["count"] == 5

    # --- Test 10: Invalid JSON handled safely ---
    @patch.object(OpenRouterClient, "complete")
    def test_invalid_json_raises_validation_error(self, mock_complete: MagicMock) -> None:
        """generate_structured should raise LLMValidationError for bad JSON."""
        mock_complete.return_value = _mock_raw_response(content="not valid json {{{")
        svc = self._make_service()

        with pytest.raises(LLMValidationError, match="not valid JSON"):
            svc.generate_structured(
                [{"role": "user", "content": "give json"}],
                retry_on_parse_failure=False,
            )

    # --- Test: Permanent error skips retries ---
    @patch.object(OpenRouterClient, "complete")
    def test_permanent_error_skips_retries(self, mock_complete: MagicMock) -> None:
        """A 401 error on primary should skip retries and go to fallback."""
        mock_complete.side_effect = [
            LLMProviderError("unauthorized", model="primary/model", status_code=401),
            _mock_raw_response(model="fallback/model"),
        ]
        svc = self._make_service()
        result = svc.generate([{"role": "user", "content": "hi"}])

        assert result.fallback_used is True
        # Should only have called primary once (no retry for 401)
        assert mock_complete.call_count == 2  # 1 primary + 1 fallback

    # --- Test: Health check returns expected content ---
    @patch.object(OpenRouterClient, "complete")
    def test_health_check(self, mock_complete: MagicMock) -> None:
        """health_check should return result with AutoDS AI OK."""
        mock_complete.return_value = _mock_raw_response(content="AutoDS AI OK")
        svc = self._make_service()
        result = svc.health_check()

        assert result.success is True
        assert "AutoDS AI OK" in result.content

    # --- Test: Tool calling interface ---
    @patch.object(OpenRouterClient, "complete")
    def test_generate_with_tools(self, mock_complete: MagicMock) -> None:
        """generate_with_tools should pass tools to the client."""
        tool_call_response = _mock_raw_response()
        tool_call_response["tool_calls"] = [
            {"id": "tc1", "type": "function", "function": {"name": "inspect_dataset", "arguments": "{}"}}
        ]
        mock_complete.return_value = tool_call_response
        svc = self._make_service()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "inspect_dataset",
                    "description": "Inspect the dataset",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = svc.generate_with_tools(
            [{"role": "user", "content": "inspect"}],
            tools=tools,
        )
        assert result.success is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "inspect_dataset"

    # --- Test: LLMResult dataclass ---
    def test_llm_result_defaults(self) -> None:
        """LLMResult should have sensible defaults."""
        r = LLMResult()
        assert r.content == ""
        assert r.success is False
        assert r.fallback_used is False
        assert r.usage == {}
        assert r.tool_calls == []
        assert r.latency_ms == 0.0
        assert r.error is None
