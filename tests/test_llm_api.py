from __future__ import annotations
import asyncio
import io
import json
from types import SimpleNamespace
from typing import cast

import httpx
import numpy as np
import pytest
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import InferenceReadiness, ModelStatus
from hcmai.llm.client import (
    InferenceClient,
    RemoteDenseEncoder,
    RemoteFrameCaptioner,
)
from hcmai.llm.config import LLMServiceConfig
from hcmai.llm.service.api import create_llm_app
from hcmai.llm.service.runtime import LLMRuntime
class FakeRuntime:
    config = LLMServiceConfig()
    reranker = SimpleNamespace(resolved_revision="test")
    captioner = SimpleNamespace(resolved_revision="caption-sha")

    def load(self):
        return None

    def readiness(self):
        return InferenceReadiness(
            ready=True, models={"visual_embedding": ModelStatus(loaded=True)}
        )

    def embed_text(self, texts, source="visual"):
        return np.asarray([[0.0, 1.0]] * len(texts), dtype=np.float32)

    def caption(self, images):
        return [f"red {image.getpixel((0, 0))[0]}" for image in images]

    def rerank(self, query, images):
        assert query == "red car"
        return [image.getpixel((0, 0))[0] / 255 for image in images]

    def resolve(self, request):
        return {
            "standalone_query": request["current_message"],
            "positive_constraints": [],
            "negative_constraints": [],
            "uncertain_constraints": [],
            "accepted_frame_ids": [],
            "rejected_frame_ids": [],
        }

    def answer_vqa(self, question, image, evidence):
        assert question == "What color?"
        assert evidence.caption == "A red square."
        return "red"
def _jpeg(red):
    output = io.BytesIO()
    Image.new("RGB", (2, 2), (red, 0, 0)).save(output, "JPEG")
    return output.getvalue()
def request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())
def test_inference_endpoints_preserve_order_and_contracts():
    app = create_llm_app(cast(LLMRuntime, FakeRuntime()))
    embedding = request(
        app,
        "POST",
        "/v1/embeddings/text",
        json={"source": "caption", "texts": ["one", "two"]},
    )
    assert embedding.status_code == 200
    assert embedding.json()["embeddings"] == [[0.0, 1.0], [0.0, 1.0]]

    captions = request(
        app, "POST", "/v1/captions",
        data={"item_ids": json.dumps(["a", "b"])},
        files=[
            ("images", ("a.jpg", _jpeg(10), "image/jpeg")),
            ("images", ("b.jpg", _jpeg(200), "image/jpeg")),
        ],
    )
    assert captions.status_code == 200
    assert [item["item_id"] for item in captions.json()["items"]] == ["a", "b"]

    rerank = request(
        app, "POST", "/v1/rerank",
        data={"query": "red car", "item_ids": json.dumps(["a", "b"])},
        files=[
            ("images", ("a.jpg", _jpeg(10), "image/jpeg")),
            ("images", ("b.jpg", _jpeg(200), "image/jpeg")),
        ],
    )
    assert rerank.status_code == 200
    assert [item["item_id"] for item in rerank.json()["items"]] == ["a", "b"]

    resolved = request(
        app, "POST", "/v1/conversation/resolve",
        json={"instruction": "resolve", "current_message": "xe đỏ"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["standalone_query"] == "xe đỏ"

    answered = request(
        app,
        "POST",
        "/v1/vqa",
        data={
            "request_id": "q1",
            "frame_id": "f1",
            "question": "What color?",
            "evidence": json.dumps({"caption": "A red square."}),
        },
        files=[("image", ("f1.jpg", _jpeg(200), "image/jpeg"))],
    )
    assert answered.status_code == 200
    assert answered.json()["answer"] == "red"
    assert answered.json()["frame_id"] == "f1"
def test_remote_encoder_validates_model_and_dimension():
    def handler(request):
        assert json.loads(request.content)["source"] == "caption"
        return httpx.Response(200, json={
            "model": "model", "dimension": 2, "normalized": True,
            "embeddings": [[0.0, 1.0]], "latency_ms": 1,
        })
    http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://model.test"
    )
    client = InferenceClient("https://model.test", client=http)
    encoder = RemoteDenseEncoder(
        client,
        EncoderConfig(model_name="model"),
        embedding_dim=2,
        source="caption",
    )
    np.testing.assert_allclose(encoder.encode_text(["query"]), [[0.0, 1.0]])


def test_remote_encoder_batches_at_api_limit():
    calls = []

    def handler(request):
        texts = json.loads(request.content)["texts"]
        calls.append(len(texts))
        return httpx.Response(200, json={
            "model": "model", "dimension": 2, "normalized": True,
            "embeddings": [[0.0, 1.0]] * len(texts), "latency_ms": 1,
        })

    http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://model.test"
    )
    encoder = RemoteDenseEncoder(
        InferenceClient("https://model.test", client=http),
        EncoderConfig(model_name="model", batch_size=100),
        embedding_dim=0,
        source="caption",
    )
    assert encoder.encode_text(["x"] * 130).shape == (130, 2)
    assert calls == [64, 64, 2]


def test_remote_captioner_validates_readiness_and_identity():
    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={
                "ready": True,
                "models": {
                    "caption_generation": {
                        "loaded": True,
                        "checkpoint": "caption/model",
                        "revision": "caption-sha",
                    }
                },
            })
        return httpx.Response(200, json={
            "model": "caption/model",
            "revision": "caption-sha",
            "items": [{"item_id": "0", "caption": "A red square."}],
            "latency_ms": 1,
        })

    http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://model.test"
    )
    config = SimpleNamespace(model_checkpoint="caption/model")
    captioner = RemoteFrameCaptioner(
        InferenceClient("https://model.test", client=http), config
    )
    assert captioner.resolve_revision() == "caption-sha"
    assert captioner.caption_batch([Image.new("RGB", (2, 2))]) == [
        "A red square."
    ]


def test_runtime_does_not_construct_or_require_disabled_models():
    runtime = LLMRuntime(
        LLMServiceConfig(),
        enable_caption=False,
        enable_visual_embedding=False,
        enable_caption_embedding=False,
        enable_reranker=False,
        enable_conversation=False,
    )

    runtime.load()
    readiness = runtime.readiness()

    assert readiness.ready is True
    assert all(not status.enabled for status in readiness.models.values())
    assert all(not status.loaded for status in readiness.models.values())
    with pytest.raises(RuntimeError, match="embedding model is disabled"):
        runtime.embed_text(["query"])
    with pytest.raises(RuntimeError, match="caption model is disabled"):
        runtime.caption([Image.new("RGB", (1, 1))])
