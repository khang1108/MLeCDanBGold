"""Contract tests for generic OpenAI-compatible text generation route."""

from __future__ import annotations

from fastapi.testclient import TestClient

from llm.server.app import create_llm_app


class FakeRuntime:
    """Fake inference runtime for testing router contracts without GPU/model loading."""

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def readiness(self):
        class Model:
            loaded = True
            checkpoint = "Qwen/Qwen3-4B"
            revision = "rev"

        class Ready:
            ready = True
            models = {"text_generation": Model()}

        return Ready()

    def generate_chat(self, messages, *, response_schema, temperature, max_tokens):
        assert messages[-1]["content"] == "resolve this"
        assert response_schema["type"] == "object"
        return '{"answer":"ok"}'


def test_chat_completions_returns_openai_shape() -> None:
    """Verify standard OpenAI chat completion envelope and content extraction."""
    client = TestClient(create_llm_app(FakeRuntime()))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-4B",
            "messages": [{"role": "user", "content": "resolve this"}],
            "temperature": 0,
            "max_tokens": 128,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "Qwen/Qwen3-4B"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == '{"answer":"ok"}'


def test_chat_completions_rejects_unready_text_model() -> None:
    """Verify HTTP 503 is returned when text_generation model is not loaded."""
    runtime = FakeRuntime()
    runtime.readiness = lambda: type(
        "Ready",
        (),
        {"models": {"text_generation": type("Model", (), {"loaded": False})()}},
    )()
    client = TestClient(create_llm_app(runtime))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-4B",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "X", "schema": {"type": "object"}},
            },
        },
    )
    assert response.status_code == 503


def test_chat_completions_rejects_model_mismatch() -> None:
    """Reject requests targeting a model checkpoint different from loaded checkpoint."""
    client = TestClient(create_llm_app(FakeRuntime()))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Different/Model",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "X", "schema": {"type": "object"}},
            },
        },
    )
    assert response.status_code == 400
