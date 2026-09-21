"""Tests for InferenceClient and InferenceClientPool embed_text integration."""

from __future__ import annotations

import json
import httpx
import pytest

from hcmai.inference.clients.embeddings import TextEmbeddingBatch
from llm.remote.client import InferenceClient
from llm.remote.pool import InferenceClientPool


def test_inference_client_embed_text_posts_to_v1_embeddings_text() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v1/embeddings/text":
            body = json.loads(request.read())
            assert body["model"] == "BAAI/bge-m3"
            assert body["input"] == ["first text", "second text"]
            return httpx.Response(
                200,
                json={
                    "model": "BAAI/bge-m3",
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                        {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                    ],
                },
            )
        raise AssertionError(f"Unexpected request: {request.url.path}")

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="https://api.test")
    client = InferenceClient("https://api.test", client=http_client)

    result = client.embed_text(["first text", "second text"], model="BAAI/bge-m3")

    assert isinstance(result, TextEmbeddingBatch)
    assert result.model == "BAAI/bge-m3"
    assert len(result.vectors) == 2
    # Verify index sorting
    assert result.vectors[0] == (1.0, 0.0, 0.0)
    assert result.vectors[1] == (0.0, 1.0, 0.0)
    assert len(calls) == 1
    assert calls[0].url.path == "/v1/embeddings/text"


def test_inference_client_pool_embed_text_delegates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/embeddings/text":
            return httpx.Response(
                200,
                json={
                    "model": "BAAI/bge-m3",
                    "data": [
                        {"index": 0, "embedding": [0.5, 0.5]},
                    ],
                },
            )
        raise AssertionError(f"Unexpected request: {request.url.path}")

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="https://api.test")
    client = InferenceClient("https://api.test", client=http_client)
    pool = InferenceClientPool(clients=[client])

    result = pool.embed_text(["sample query"], model="BAAI/bge-m3")
    assert isinstance(result, TextEmbeddingBatch)
    assert result.model == "BAAI/bge-m3"
    assert result.vectors == ((0.5, 0.5),)
