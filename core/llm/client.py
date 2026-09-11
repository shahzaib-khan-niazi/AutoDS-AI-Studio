"""OpenRouter client wrapper using OpenAI-compatible SDK."""

import re
from typing import Any, Optional
from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError

from core.logging import logger
from core.llm.config import LLMConfig, llm_config
from core.llm.exceptions import LLMProviderError
from core.utils.json_sanitizer import sanitize_for_json


class OpenRouterClient:
    """Wrapper around OpenAI SDK targeting OpenRouter."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or llm_config
        self._client: Optional[OpenAI] = None

    def get_client(self) -> OpenAI:
        """Get or initialize OpenAI client for OpenRouter."""
        self.config.validate()

        if self._client is None:
            default_headers = {
                "HTTP-Referer": self.config.app_url,
                "X-Title": self.config.app_name,
            }
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                default_headers=default_headers,
            )
        return self._client

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        reasoning: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        """Send chat completion request to a specific model.

        Args:
            model: Target model ID (e.g. nvidia/nemotron-3-super-120b-a12b:free).
            messages: List of message dictionaries.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            response_format: Response format (e.g. {"type": "json_object"}).
            tools: Optional tool definitions.
            tool_choice: Tool choice parameter.
            reasoning: Model-specific reasoning configuration.
            timeout: Request timeout override.

        Returns:
            Dictionary with content, finish_reason, usage, tool_calls, raw_response.

        Raises:
            LLMProviderError: On any provider error, rate limit, timeout, or network issue.
        """
        client = self.get_client()

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": sanitize_for_json(messages),
            "temperature": temperature,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if response_format is not None:
            kwargs["response_format"] = sanitize_for_json(response_format)

        if tools is not None:
            kwargs["tools"] = sanitize_for_json(tools)
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        if reasoning is not None:
            # Pass reasoning config if supported/requested
            kwargs["extra_body"] = {"reasoning": sanitize_for_json(reasoning)}

        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            response = client.chat.completions.create(**kwargs)

            choices = getattr(response, "choices", None)
            if not choices:
                raise LLMProviderError(
                    message=f"Empty response from model '{model}' (no choices returned)",
                    model=model,
                    status_code=None,
                )

            choice = choices[0]
            raw_content = getattr(choice.message, "content", "") or ""
            finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

            # Sanitize any inline <thought> or <reasoning> tags so private traces are not exposed
            clean_content = self._sanitize_content(raw_content)

            # Tool calls extraction if any
            tool_calls = []
            tc_list = getattr(choice.message, "tool_calls", None)
            if tc_list:
                for tc in tc_list:
                    fn_obj = getattr(tc, "function", None)
                    fn_name = getattr(fn_obj, "name", "") if fn_obj else ""
                    fn_args = getattr(fn_obj, "arguments", "") if fn_obj else ""
                    tool_calls.append({
                        "id": getattr(tc, "id", ""),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": fn_name,
                            "arguments": fn_args,
                        },
                    })

            usage_dict = {}
            if getattr(response, "usage", None):
                usage_dict = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

            return {
                "content": clean_content,
                "finish_reason": finish_reason,
                "usage": usage_dict,
                "tool_calls": tool_calls,
                "model_used": getattr(response, "model", model),
            }

        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            status_code = getattr(e, "status_code", None)
            err_msg = str(e)
            logger.warning("Provider error with model='{}' [status={}]: {}", model, status_code, err_msg)
            raise LLMProviderError(
                message=f"Model provider '{model}' call failed: {err_msg}",
                model=model,
                status_code=status_code,
                details=err_msg,
            ) from e
        except Exception as e:
            logger.warning("Unexpected error with model='{}': {}", model, str(e))
            raise LLMProviderError(
                message=f"Unexpected error calling '{model}': {str(e)}",
                model=model,
                details=str(e),
            ) from e

    @staticmethod
    def _sanitize_content(content: str) -> str:
        """Remove private thought/reasoning tags from final content."""
        if not content:
            return ""
        # Strip <think>...</think> or <thought>...</thought> tags, including truncated tags
        cleaned = re.sub(r"<(think|thought)>.*?(?:</\1>|$)", "", content, flags=re.DOTALL)
        return cleaned.strip()
