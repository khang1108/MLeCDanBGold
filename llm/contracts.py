"""Private inference-service HTTP contracts with no HCMAI runtime ownership."""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


_NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class _HTTPContract(BaseModel):
    """Reject extra fields on private inference requests and responses."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class BoundaryScoreResponse(_HTTPContract):
    """Per-frame scores returned by a private boundary model."""

    request_id: _NonEmptyString
    model: _NonEmptyString
    revision: str | None = None
    scores: list[float] = Field(min_length=1)
    latency_ms: float = Field(ge=0)


class ChatMessage(_HTTPContract):
    """One conversation message formatted for chat completion."""

    role: Literal["system", "user", "assistant"]
    content: _NonEmptyString


class JsonSchemaSpec(_HTTPContract):
    """JSON schema definition for structured output."""

    name: _NonEmptyString
    schema_: dict[str, Any] = Field(alias="schema")


class ResponseFormat(_HTTPContract):
    """Structured response format specification."""

    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec


class ChatCompletionRequest(_HTTPContract):
    """OpenAI-compatible chat completion request."""

    model: _NonEmptyString
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    response_format: ResponseFormat


class ChatChoiceMessage(_HTTPContract):
    """Assistant message in a completion choice."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(_HTTPContract):
    """One candidate choice in an OpenAI completion response."""

    index: int = 0
    message: ChatChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(_HTTPContract):
    """OpenAI-compatible chat completion response envelope."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]


__all__ = [
    "BoundaryScoreResponse",
    "ChatMessage",
    "JsonSchemaSpec",
    "ResponseFormat",
    "ChatCompletionRequest",
    "ChatChoiceMessage",
    "ChatChoice",
    "ChatCompletionResponse",
]
