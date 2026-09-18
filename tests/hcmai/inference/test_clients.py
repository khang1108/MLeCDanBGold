"""Contract tests for inference clients, model identity, and error normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pytest
from pydantic import BaseModel

from hcmai.inference.config import ModelEndpointConfig
from hcmai.inference.errors import (
    InferenceAuthError,
    InferenceError,
    InferenceResponseError,
    InferenceUnavailableError,
)
from hcmai.inference.embeddings import EmbeddingClient
from hcmai.inference.http import HttpTransport
from hcmai.inference.llm import LLMClient


@dataclass
class RecordedCall:
    url: str
    payload: dict[str, Any]
    headers: dict[str, str]
    timeout: float


class FakeTransport(HttpTransport):
    """Fake transport recording calls and returning queued responses or raising exceptions."""

    def __init__(self, response_data: Any = None, exc: Exception | None = None) -> None:
        self.calls: list[RecordedCall] = []
        self.response_data = response_data
        self.exc = exc

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        self.calls.append(RecordedCall(url=url, payload=payload, headers=headers, timeout=timeout))
        if self.exc is not None:
            raise self.exc
        return self.response_data or {}


class SampleResponse(BaseModel):
    message: str


def test_llm_client_appends_chat_completions_and_exposes_model() -> None:
    """Verify LLMClient posts to /chat/completions and exposes the model property."""
    endpoint = ModelEndpointConfig(
        base_url="https://api.example/v1",
        model="Qwen/Qwen3-4B",
        timeout_seconds=30.0,
    )
    transport = FakeTransport(
        response_data={
            "choices": [
                {"message": {"role": "assistant", "content": '{"message": "hello"}'}}
            ]
        }
    )
    client = LLMClient(endpoint, transport=transport)

    assert client.model == "Qwen/Qwen3-4B"

    result = client.generate_structured(
        [{"role": "user", "content": "hi"}],
        SampleResponse,
    )
    assert result.message == "hello"
    assert len(transport.calls) == 1
    assert transport.calls[0].url == "https://api.example/v1/chat/completions"


def test_llm_client_preserves_json_schema_payload() -> None:
    """Verify LLMClient serializes OpenAI json_schema response_format correctly."""
    endpoint = ModelEndpointConfig(
        base_url="https://api.example/v1",
        model="Qwen/Qwen3-4B",
        enable_thinking=False,
        max_tokens=4096,
    )
    transport = FakeTransport(
        response_data={
            "choices": [
                {"message": {"role": "assistant", "content": '{"message": "tested"}'}}
            ]
        }
    )
    client = LLMClient(endpoint, transport=transport)
    client.generate_structured([{"role": "user", "content": "hi"}], SampleResponse)

    assert len(transport.calls) == 1
    payload = transport.calls[0].payload
    assert payload["model"] == "Qwen/Qwen3-4B"
    assert payload["max_tokens"] == 4096
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "SampleResponse"
    assert "properties" in payload["response_format"]["json_schema"]["schema"]


def test_llm_client_reports_length_exceeded_clearly() -> None:
    """Verify finish_reason='length' produces a clear, actionable error."""
    endpoint = ModelEndpointConfig(base_url="https://api.example/v1", model="Qwen/Qwen3-4B")
    transport = FakeTransport(
        response_data={
            "choices": [
                {"finish_reason": "length", "message": {"content": "", "reasoning_content": "rambling..."}}
            ]
        }
    )
    client = LLMClient(endpoint, transport=transport)
    with pytest.raises(InferenceResponseError) as exc_info:
        client.generate_structured([{"role": "user", "content": "hi"}], SampleResponse)
    assert "finish_reason='length'" in str(exc_info.value)


def test_llm_client_maps_invalid_content_to_response_error() -> None:
    """Verify malformed JSON or schema violation raises InferenceResponseError."""
    endpoint = ModelEndpointConfig(base_url="https://api.example/v1", model="Qwen/Qwen3-4B")

    # Non-JSON content
    transport_bad_json = FakeTransport(
        response_data={"choices": [{"message": {"content": "not json"}}]}
    )
    client = LLMClient(endpoint, transport=transport_bad_json)
    with pytest.raises(InferenceResponseError) as exc_info:
        client.generate_structured([{"role": "user", "content": "hi"}], SampleResponse)
    assert exc_info.value.__cause__ is not None

    # Schema mismatch
    transport_bad_schema = FakeTransport(
        response_data={"choices": [{"message": {"content": '{"other": 123}'}}]}
    )
    client2 = LLMClient(endpoint, transport=transport_bad_schema)
    with pytest.raises(InferenceResponseError) as exc_info:
        client2.generate_structured([{"role": "user", "content": "hi"}], SampleResponse)
    assert exc_info.value.__cause__ is not None

    # Empty choices
    transport_empty = FakeTransport(response_data={"choices": []})
    client3 = LLMClient(endpoint, transport=transport_empty)
    with pytest.raises(InferenceResponseError):
        client3.generate_structured([{"role": "user", "content": "hi"}], SampleResponse)


def test_embedding_client_appends_embeddings_and_preserves_input_order() -> None:
    """Verify EmbeddingClient posts to /embeddings, exposes model, and preserves input order."""
    endpoint = ModelEndpointConfig(
        base_url="https://api.example/v1",
        model="BAAI/bge-m3",
        timeout_seconds=15.0,
    )
    transport = FakeTransport(
        response_data={
            "model": "BAAI/bge-m3",
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ],
        }
    )
    client = EmbeddingClient(endpoint, transport=transport)

    assert client.model == "BAAI/bge-m3"

    batch = client.embed_text(["first", "second"])
    assert len(transport.calls) == 1
    assert transport.calls[0].url == "https://api.example/v1/embeddings"
    assert transport.calls[0].payload["input"] == ["first", "second"]
    assert batch.vectors == ((0.1, 0.2), (0.3, 0.4))


def test_embedding_client_maps_malformed_response_to_response_error() -> None:
    """Verify mismatched count or invalid indices raise InferenceResponseError."""
    endpoint = ModelEndpointConfig(base_url="https://api.example/v1", model="BAAI/bge-m3")

    # Mismatched count
    transport_mismatch = FakeTransport(
        response_data={"data": [{"index": 0, "embedding": [0.1]}]}
    )
    client = EmbeddingClient(endpoint, transport=transport_mismatch)
    with pytest.raises(InferenceResponseError):
        client.embed_text(["first", "second"])

    # Invalid indices
    transport_bad_indices = FakeTransport(
        response_data={
            "data": [
                {"index": 0, "embedding": [0.1]},
                {"index": 0, "embedding": [0.2]},
            ]
        }
    )
    client2 = EmbeddingClient(endpoint, transport=transport_bad_indices)
    with pytest.raises(InferenceResponseError):
        client2.embed_text(["first", "second"])


def test_auth_failure_is_not_reported_as_unavailable() -> None:
    """Verify HTTP 401/403 maps to InferenceAuthError, not InferenceUnavailableError."""
    endpoint = ModelEndpointConfig(base_url="https://api.example/v1", model="Qwen/Qwen3-4B")
    transport = FakeTransport(exc=InferenceAuthError("Invalid API key"))
    client = LLMClient(endpoint, transport=transport)

    with pytest.raises(InferenceAuthError):
        client.generate_structured([{"role": "user", "content": "hi"}], SampleResponse)


def test_http_transport_translates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify HttpTransport maps low-level httpx errors to typed taxonomy."""
    import httpx

    transport = HttpTransport()

    # Network / Timeout -> InferenceUnavailableError
    def mock_post_timeout(*args: Any, **kwargs: Any) -> Any:
        raise httpx.ConnectTimeout("connection timed out")

    monkeypatch.setattr(httpx.Client, "post", mock_post_timeout)
    with pytest.raises(InferenceUnavailableError):
        transport.post_json("http://example.test", {}, {}, 1.0)

    # 401 Auth -> InferenceAuthError
    mock_resp_401 = httpx.Response(401, text="Unauthorized", request=httpx.Request("POST", "http://example.test"))
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: mock_resp_401)
    with pytest.raises(InferenceAuthError):
        transport.post_json("http://example.test", {}, {}, 1.0)

    # 500 Server Error -> InferenceResponseError
    mock_resp_500 = httpx.Response(500, text="Internal Server Error", request=httpx.Request("POST", "http://example.test"))
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: mock_resp_500)
    with pytest.raises(InferenceResponseError):
        transport.post_json("http://example.test", {}, {}, 1.0)

    # 200 but invalid JSON -> InferenceResponseError
    mock_resp_bad_json = httpx.Response(200, text="not json <xml>", request=httpx.Request("POST", "http://example.test"))
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: mock_resp_bad_json)
    with pytest.raises(InferenceResponseError):
        transport.post_json("http://example.test", {}, {}, 1.0)


def test_scoped_resolution_without_language_parses_from_raw_transport() -> None:
    """Verify raw model payload without language validates as ScopedResolutionBatch."""
    from hcmai.kis.resolution import ScopedResolutionBatch

    raw_json = '{"events":[{"event_id":"E1","text":"Con chó được cập nhật","bindings":[]}]}'
    endpoint = ModelEndpointConfig(base_url="https://api.example/v1", model="Qwen/Qwen3-4B")
    transport = FakeTransport(
        response_data={
            "choices": [
                {"message": {"role": "assistant", "content": raw_json}}
            ]
        }
    )
    client = LLMClient(endpoint, transport=transport)
    result = client.generate_structured([{"role": "user", "content": "patch"}], ScopedResolutionBatch)
    assert len(result.events) == 1
    assert result.events[0].event_id == "E1"
    assert result.events[0].text == "Con chó được cập nhật"

