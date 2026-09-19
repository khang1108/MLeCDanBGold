from __future__ import annotations
import time
from typing import Any, Literal
import uuid
from pydantic import Field
from .common import HTTPContract, NonEmptyString

class BoundaryScoreResponse(HTTPContract):
    request_id: NonEmptyString
    model: NonEmptyString
    revision: str | None = None
    scores: list[float] = Field(min_length=1)
    latency_ms: float = Field(ge=0)

class ChatMessage(HTTPContract):
    role: Literal["system", "user", "assistant"]
    content: NonEmptyString

class JsonSchemaSpec(HTTPContract):
    name: NonEmptyString
    schema_: dict[str, Any] = Field(alias="schema")

class ResponseFormat(HTTPContract):
    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec

class ChatCompletionRequest(HTTPContract):
    model: NonEmptyString
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    response_format: ResponseFormat

class ChatChoiceMessage(HTTPContract):
    role: Literal["assistant"] = "assistant"
    content: str

class ChatChoice(HTTPContract):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(HTTPContract):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
