"""Tests for provider-agnostic inference configuration and clients."""

import os
import unittest
from unittest.mock import Mock, patch

from pydantic import BaseModel, ValidationError

from hcmai.inference.config import (
    ModelEndpointConfig,
    load_embedding_endpoint,
    load_llm_endpoint,
)
from hcmai.inference.embeddings import (
    EmbeddingClient,
    OpenAICompatibleEmbeddingClient,
    TextEmbeddingBatch,
)
from hcmai.inference.http import HttpTransport
from hcmai.inference.llm import LLMClient, OpenAICompatibleLLMClient


class Sample(BaseModel):
    value: str


class InferenceConfigTest(unittest.TestCase):
    def test_loads_llm_endpoint_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HCMAI_LLM_BASE_URL": "https://api.example/v1",
                "HCMAI_LLM_API_KEY": "secret",
                "HCMAI_LLM_MODEL": "qwen3-4b",
                "HCMAI_LLM_TIMEOUT_SECONDS": "17",
            },
            clear=False,
        ):
            config = load_llm_endpoint()
        self.assertEqual(config.base_url, "https://api.example/v1")
        self.assertEqual(config.api_key, "secret")
        self.assertEqual(config.model, "qwen3-4b")
        self.assertEqual(config.timeout_seconds, 17)

    def test_loads_embedding_endpoint_independently(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HCMAI_EMBEDDING_BASE_URL": "https://embed.example/v1",
                "HCMAI_EMBEDDING_MODEL": "bge-m3",
            },
            clear=False,
        ):
            config = load_embedding_endpoint()
        self.assertEqual(config.base_url, "https://embed.example/v1")
        self.assertIsNone(config.api_key)
        self.assertEqual(config.model, "bge-m3")
        self.assertEqual(config.timeout_seconds, 30.0)

    def test_rejects_blank_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HCMAI_LLM_BASE_URL": "https://api.example/v1",
                "HCMAI_LLM_MODEL": "  ",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "must not be blank"):
                load_llm_endpoint()

    def test_rejects_non_positive_timeout(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HCMAI_LLM_BASE_URL": "https://api.example/v1",
                "HCMAI_LLM_MODEL": "qwen3-4b",
                "HCMAI_LLM_TIMEOUT_SECONDS": "0",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                load_llm_endpoint()


class LLMClientTest(unittest.TestCase):
    def test_alias_is_identical(self) -> None:
        self.assertIs(OpenAICompatibleLLMClient, LLMClient)

    def test_structured_generation_validates_response_model(self) -> None:
        transport = Mock()
        transport.post_json.return_value = {
            "choices": [{"message": {"content": '{"value":"ok"}'}}]
        }
        client = LLMClient(
            ModelEndpointConfig("https://x/v1", "test-key", "test-model", 10),
            transport=transport,
        )
        result = client.generate_structured(
            [{"role": "user", "content": "test"}], Sample
        )
        self.assertEqual(result, Sample(value="ok"))

        transport.post_json.assert_called_once()
        url, payload, headers, timeout = transport.post_json.call_args[0]
        self.assertEqual(url, "https://x/v1/chat/completions")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "test"}])
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["response_format"]["type"], "json_object")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(timeout, 10)

    def test_structured_generation_omits_auth_header_when_api_key_none(self) -> None:
        transport = Mock()
        transport.post_json.return_value = {
            "choices": [{"message": {"content": '{"value":"hello"}'}}]
        }
        client = LLMClient(
            ModelEndpointConfig("https://x/v1", None, "test-model", 10),
            transport=transport,
        )
        result = client.generate_structured([{"role": "user", "content": "hi"}], Sample)
        self.assertEqual(result, Sample(value="hello"))

        _, _, headers, _ = transport.post_json.call_args[0]
        self.assertNotIn("Authorization", headers)

    def test_structured_generation_raises_on_malformed_json(self) -> None:
        transport = Mock()
        transport.post_json.return_value = {
            "choices": [{"message": {"content": "not-a-json"}}]
        }
        client = LLMClient(
            ModelEndpointConfig("https://x/v1", None, "test-model", 10),
            transport=transport,
        )
        with self.assertRaises(ValueError):
            client.generate_structured([{"role": "user", "content": "hi"}], Sample)

    def test_structured_generation_raises_on_schema_validation_error(self) -> None:
        transport = Mock()
        transport.post_json.return_value = {
            "choices": [{"message": {"content": '{"wrong_key": 123}'}}]
        }
        client = LLMClient(
            ModelEndpointConfig("https://x/v1", None, "test-model", 10),
            transport=transport,
        )
        with self.assertRaises(ValidationError):
            client.generate_structured([{"role": "user", "content": "hi"}], Sample)

    def test_structured_generation_raises_on_empty_choices(self) -> None:
        transport = Mock()
        transport.post_json.return_value = {"choices": []}
        client = LLMClient(
            ModelEndpointConfig("https://x/v1", None, "test-model", 10),
            transport=transport,
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            client.generate_structured([{"role": "user", "content": "hi"}], Sample)


class EmbeddingClientTest(unittest.TestCase):
    def test_alias_is_identical(self) -> None:
        self.assertIs(OpenAICompatibleEmbeddingClient, EmbeddingClient)

    def test_embedding_client_preserves_input_order(self) -> None:
        transport = Mock()
        transport.post_json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ],
            "model": "embed",
        }
        client = EmbeddingClient(
            ModelEndpointConfig("https://x/v1", "embed-key", "embed", 10),
            transport=transport,
        )
        batch = client.embed_text(["a", "b"])
        self.assertEqual(batch.model, "embed")
        self.assertEqual(batch.vectors, ((1.0, 0.0), (0.0, 1.0)))

        transport.post_json.assert_called_once()
        url, payload, headers, timeout = transport.post_json.call_args[0]
        self.assertEqual(url, "https://x/v1/embeddings")
        self.assertEqual(payload["model"], "embed")
        self.assertEqual(payload["input"], ["a", "b"])
        self.assertEqual(headers["Authorization"], "Bearer embed-key")
        self.assertEqual(timeout, 10)

    def test_embedding_client_raises_on_count_mismatch(self) -> None:
        transport = Mock()
        transport.post_json.return_value = {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
            ],
            "model": "embed",
        }
        client = EmbeddingClient(
            ModelEndpointConfig("https://x/v1", None, "embed", 10),
            transport=transport,
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            client.embed_text(["a", "b"])


if __name__ == "__main__":
    unittest.main()
