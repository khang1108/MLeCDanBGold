"""Provider-agnostic LLM client capability and OpenAI-compatible implementation.

This module owns structured text generation through the LLMClient.
It does not hardcode provider names or prompts; callers pass their own message
sequences and response schemas.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any, TypeVar

from pydantic import BaseModel

from hcmai.inference.config import ModelEndpointConfig
from hcmai.inference.errors import InferenceResponseError
from hcmai.inference.http import HttpTransport

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """HTTP client for structured text generation using remote LLM endpoints."""

    def __init__(
        self,
        endpoint: ModelEndpointConfig,
        transport: HttpTransport | None = None,
    ) -> None:
        """Initialize the client with endpoint configuration and transport."""
        self._endpoint = endpoint
        self._transport = transport or HttpTransport()

    @property
    def model(self) -> str:
        """Return the configured model checkpoint or identifier."""
        return self._endpoint.model

    def generate_structured(
        self,
        messages: Sequence[dict[str, str]],
        response_model: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> T:
        """Send the response schema and validate the returned JSON locally.

        The endpoint must support the OpenAI-compatible json_schema response
        format. Local validation also enforces domain model validators that
        cannot be represented in JSON Schema.

        Args:
            messages: OpenAI-style list of message dictionaries with role and content.
            response_model: The Pydantic model type to validate against.
            temperature: Sampling temperature (default 0.0 for deterministic outputs).
            max_tokens: Maximum tokens in response.

        Returns:
            An instance of response_model parsed from the LLM JSON output.

        Raises:
            InferenceResponseError: If completion choices are empty, content cannot be
                parsed as JSON, or schema validation fails.
        """
        url = f"{self._endpoint.base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._endpoint.api_key:
            headers["Authorization"] = f"Bearer {self._endpoint.api_key}"

        tokens_budget = max_tokens or self._endpoint.max_tokens or 4096
        payload: dict[str, Any] = {
            "model": self._endpoint.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": tokens_budget,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            },
        }
        if self._endpoint.enable_thinking is not None and "groq.com" not in self._endpoint.base_url.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": self._endpoint.enable_thinking}

        data = self._transport.post_json(
            url,
            payload,
            headers,
            self._endpoint.timeout_seconds,
        )

        choices = data.get("choices")
        if not choices:
            raise InferenceResponseError(f"OpenAI completion returned empty choices: {data}")

        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if not content:
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                raise InferenceResponseError(
                    f"OpenAI completion exceeded max_tokens limit ({tokens_budget}) before producing content "
                    f"(finish_reason='length', model={self._endpoint.model!r}). "
                    "Set HCMAI_LLM_ENABLE_THINKING=false or increase HCMAI_LLM_MAX_TOKENS."
                )
            raise InferenceResponseError(f"OpenAI completion message content was empty: {data}")

        try:
            parsed = json.loads(content)
        except Exception as exc:
            raise InferenceResponseError(f"Failed to parse LLM response as JSON: {content}") from exc

        try:
            return response_model.model_validate(parsed)
        except Exception as exc:
            raise InferenceResponseError(f"LLM response failed schema validation: {exc}") from exc
