from __future__ import annotations
import asyncio
import io
import json
from types import SimpleNamespace

import httpx
import numpy as np
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import InferenceReadiness, ModelStatus
from hcmai.llm.api import create_llm_app
from hcmai.llm.client import InferenceClient, RemoteDenseEncoder
from hcmai.llm.config import LLMServiceConfig
class FakeRuntime:
    config = LLMServiceConfig()
    reranker = SimpleNamespace(resolved_revision="test")

    def load(self):
        return None

    def readiness(self):
        return InferenceReadiness(
            ready=True, models={"embedding": ModelStatus(loaded=True)}
        )

    def embed_text(self, texts):
        return np.asarray([[0.0, 1.0]] * len(texts), dtype=np.float32)

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
    app = create_llm_app(FakeRuntime())
    embedding = request(
        app, "POST", "/v1/embeddings/text", json={"texts": ["one", "two"]}
    )
    assert embedding.status_code == 200
    assert embedding.json()["embeddings"] == [[0.0, 1.0], [0.0, 1.0]]

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
def test_remote_encoder_validates_model_and_dimension():
    def handler(_):
        return httpx.Response(200, json={
            "model": "model", "dimension": 2, "normalized": True,
            "embeddings": [[0.0, 1.0]], "latency_ms": 1,
        })
    http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://model.test"
    )
    client = InferenceClient("https://model.test", client=http)
    encoder = RemoteDenseEncoder(
        client, EncoderConfig(model_name="model"), embedding_dim=2
    )
    np.testing.assert_allclose(encoder.encode_text(["query"]), [[0.0, 1.0]])
