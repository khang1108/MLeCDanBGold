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
from hcmai.common.schemas import (
    AudioReferenceRequest,
    DiarizationRequest,
    InferenceReadiness,
    ModelStatus,
    TranscriptSegment,
    VQAInferenceResponse,
)
from thundercompute.adapters.http import InferenceClient
from thundercompute.config import LLMServiceConfig
from hcmai.retrieval.embedding.adapters.remote import RemoteEmbeddingAdapter
from hcmai.data.enrichment.caption.adapters.remote import RemoteCaptionAdapter
from hcmai.data.enrichment.ocr.adapters.florence import FlorenceAdapter
from hcmai.data.enrichment.ocr.config import OCRConfig
from hcmai.data.enrichment.ocr.models.entities import OCRRegionResult, OCRResult
from thundercompute.adapters.local import LocalAdapter
from thundercompute.pipeline import LLMService
from thundercompute.server.api import create_llm_app
class FakeRuntime:
    config = LLMServiceConfig.from_yaml("thundercompute/config.yaml")
    reranker = SimpleNamespace(resolved_revision="test")
    captioner = SimpleNamespace(resolved_revision="caption-sha")

    def load(self):
        return None

    def readiness(self):
        return InferenceReadiness(
            ready=True,
            models={
                "visual_embedding": ModelStatus(
                    loaded=True, checkpoint="visual/model", revision="visual-sha"
                ),
                "dino": ModelStatus(
                    loaded=True, checkpoint="dino/model", revision="dino-sha"
                ),
                "ocr": ModelStatus(
                    loaded=True, checkpoint="ocr/model", revision="ocr-sha"
                ),
                "transnet": ModelStatus(
                    loaded=True, checkpoint="transnet", revision="transnet-sha"
                ),
                "efficientgebd": ModelStatus(
                    loaded=True, checkpoint="gebd", revision="gebd-sha"
                ),
                "asr": ModelStatus(
                    loaded=True, checkpoint="asr/model", revision="asr-sha"
                ),
                "diarization": ModelStatus(
                    loaded=True, checkpoint="diar/model", revision="diar-sha"
                ),
            },
        )

    def embed_text(self, texts, source="visual"):
        return np.asarray([[0.0, 1.0]] * len(texts), dtype=np.float32)

    def caption(self, images):
        return [f"red {image.getpixel((0, 0))[0]}" for image in images]

    def ocr(self, images):
        return [
            OCRResult(
                text=f"text {image.getpixel((0, 0))[0]}",
                raw_output=(
                    object()
                    if image.getpixel((0, 0))[0] < 100
                    else np.asarray([1, 2], dtype=np.int64)
                ),
                regions=(
                    OCRRegionResult(
                        text=f"text {image.getpixel((0, 0))[0]}",
                        confidence=None,
                        x_min=0.0,
                        y_min=0.0,
                        x_max=1.0,
                        y_max=1.0,
                    ),
                ),
            )
            for image in images
        ]

    def embed_images(self, images, source="visual"):
        assert source in {"visual", "dino"}
        return np.asarray([[0.0, 1.0]] * len(images), dtype=np.float32)

    def boundary_scores(self, frames, source="shot"):
        assert source in {"shot", "event"}
        return np.linspace(0, 1, len(frames), dtype=np.float32)

    def transcribe_reference(self, payload):
        return [_segment(payload.video_id)]

    def diarize_reference(self, payload):
        return [segment.model_copy(update={"speaker_id": "SPEAKER_00"}) for segment in payload.segments]

    def rerank(self, query, images):
        assert query == "red car"
        return [image.getpixel((0, 0))[0] / 255 for image in images]

    def answer_vqa(self, question, image, evidence, *, scene_context=""):
        assert question == "What color?"
        assert evidence.caption == "A red square."
        assert scene_context == "red square scene"
        return "red"

    def answer_vqa_multi(
        self, question, images, frame_ids, evidence, *, scene_context=""
    ):
        assert question == "What color?"
        assert frame_ids == ["f1", "f2"]
        assert len(images) == 2
        assert scene_context == "red square scene"
        return {
            "answer": "red",
            "selected_frame_id": "f2",
            "answerable": True,
            "confidence": 0.9,
        }
def _jpeg(red):
    output = io.BytesIO()
    Image.new("RGB", (2, 2), (red, 0, 0)).save(output, "JPEG")
    return output.getvalue()


def _segment(video_id="video-1"):
    return TranscriptSegment(
        segment_id=f"{video_id}_segment_000000",
        video_id=video_id,
        segment_index=0,
        start_ms=0,
        end_ms=1000,
        text="xin chao",
        language="vi",
    )


def _npy(value):
    output = io.BytesIO()
    np.save(output, value, allow_pickle=False)
    return output.getvalue()
def request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(send())
    finally:
        loop.close()


def test_inference_endpoints_preserve_order_and_contracts():
    app = create_llm_app(cast(LLMService, FakeRuntime()))
    embedding = request(
        app,
        "POST",
        "/v1/embeddings/text",
        json={"source": "text", "texts": ["one", "two"]},
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

    assert request(app, "POST", "/v1/query-suggestions", json={}).status_code == 404

    answered = request(
        app,
        "POST",
        "/v1/vqa",
        data={
            "request_id": "q1",
            "frame_id": "f1",
            "video_id": "video-1",
            "scene_context": "red square scene",
            "question": "What color?",
            "evidence": json.dumps({"caption": "A red square."}),
        },
        files=[("image", ("f1.jpg", _jpeg(200), "image/jpeg"))],
    )
    assert answered.status_code == 200
    assert answered.json()["answer"] == "red"
    assert answered.json()["frame_ids"] == ["f1"]
    assert answered.json()["selected_frame_id"] == "f1"
    assert answered.json()["video_id"] == "video-1"

    multi = request(
        app,
        "POST",
        "/v1/vqa/multi",
        data={
            "request_id": "q2",
            "video_id": "video-1",
            "frame_ids": json.dumps(["f1", "f2"]),
            "scene_context": "red square scene",
            "question": "What color?",
        },
        files=[
            ("images", ("f1.jpg", _jpeg(10), "image/jpeg")),
            ("images", ("f2.jpg", _jpeg(200), "image/jpeg")),
        ],
    )
    assert multi.status_code == 200
    assert multi.json()["video_id"] == "video-1"
    assert multi.json()["frame_ids"] == ["f1", "f2"]
    assert multi.json()["selected_frame_id"] == "f2"


def test_embedding_endpoints_use_configured_batch_ceilings():
    """Visual and BGE endpoints accept their own configured request sizes."""

    runtime = FakeRuntime()
    runtime.config = runtime.config.model_copy(
        update={
            "visual_embedding": runtime.config.visual_embedding.model_copy(
                update={"batch_size": 128}
            ),
            "caption_embedding": runtime.config.caption_embedding.model_copy(
                update={"batch_size": 96}
            ),
        }
    )
    app = create_llm_app(cast(LLMService, runtime))

    text = request(
        app,
        "POST",
        "/v1/embeddings/text",
        json={"source": "text", "texts": ["caption"] * 96},
    )
    assert text.status_code == 200
    assert len(text.json()["embeddings"]) == 96

    text_over_limit = request(
        app,
        "POST",
        "/v1/embeddings/text",
        json={"source": "text", "texts": ["caption"] * 97},
    )
    assert text_over_limit.status_code == 422
    assert "1..96" in text_over_limit.json()["detail"]

    def image_payload(count: int):
        return {
            "data": {
                "item_ids": json.dumps(
                    [str(index) for index in range(count)]
                )
            },
            "files": [
                (
                    "images",
                    (f"{index}.jpg", _jpeg(index), "image/jpeg"),
                )
                for index in range(count)
            ],
        }

    images = request(
        app,
        "POST",
        "/v1/embeddings/images",
        **image_payload(128),
    )
    assert images.status_code == 200
    assert len(images.json()["item_ids"]) == 128

    images_over_limit = request(
        app,
        "POST",
        "/v1/embeddings/images",
        **image_payload(129),
    )
    assert images_over_limit.status_code == 400
    assert "1..128" in images_over_limit.json()["detail"]


def test_remote_vqa_endpoints_share_one_inference_response_contract():
    def handler(request):
        is_multi = request.url.path.endswith("/multi")
        return httpx.Response(200, json={
            "request_id": "q2" if is_multi else "q1",
            "video_id": "video-1",
            "frame_ids": ["f1", "f2"] if is_multi else ["f1"],
            "selected_frame_id": "f2" if is_multi else "f1",
            "question": "What color?",
            "answer": "red",
            "answerable": True,
            "grounded": True,
            "confidence": 0.9,
            "latency_ms": 1,
        })

    http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://model.test"
    )
    client = InferenceClient("https://model.test", client=http)
    one = client.answer_vqa(
        "q1",
        "f1",
        "video-1",
        "What color?",
        Image.new("RGB", (2, 2), "red"),
    )
    many = client.answer_vqa_multi(
        "q2",
        "video-1",
        ["f1", "f2"],
        "What color?",
        [Image.new("RGB", (2, 2), "blue"), Image.new("RGB", (2, 2), "red")],
    )

    assert isinstance(one, VQAInferenceResponse)
    assert isinstance(many, VQAInferenceResponse)
    assert one.frame_ids == ["f1"]
    assert many.selected_frame_id == "f2"


def test_remote_encoder_validates_model_and_dimension():
    def handler(request):
        assert json.loads(request.content)["source"] == "text"
        return httpx.Response(200, json={
            "model": "model", "dimension": 2, "normalized": True,
            "embeddings": [[0.0, 1.0]], "latency_ms": 1,
        })
    http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://model.test"
    )
    client = InferenceClient("https://model.test", client=http)
    encoder = RemoteEmbeddingAdapter(
        client,
        EncoderConfig(model_name="model"),
        embedding_dim=2,
        source="text",
    )
    np.testing.assert_allclose(encoder.encode_text(["query"]), [[0.0, 1.0]])


def test_remote_encoder_batches_at_configured_limit():
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
    encoder = RemoteEmbeddingAdapter(
        InferenceClient("https://model.test", client=http),
        EncoderConfig(model_name="model", batch_size=128),
        embedding_dim=0,
        source="text",
    )
    assert encoder.encode_text(["x"] * 130).shape == (130, 2)
    assert calls == [128, 2]


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
    captioner = RemoteCaptionAdapter(
        InferenceClient("https://model.test", client=http), config
    )
    assert captioner.resolve_revision() == "caption-sha"
    assert captioner.caption_batch([Image.new("RGB", (2, 2))]) == [
        "A red square."
    ]


def test_offline_inference_endpoints_preserve_identity_and_provenance():
    app = create_llm_app(cast(LLMService, FakeRuntime()))
    files = [
        ("images", ("a.jpg", _jpeg(10), "image/jpeg")),
        ("images", ("b.jpg", _jpeg(200), "image/jpeg")),
    ]

    ocr = request(
        app,
        "POST",
        "/v1/enrichment/ocr",
        data={"item_ids": json.dumps(["a", "b"])},
        files=files,
    )
    assert ocr.status_code == 200
    assert [item["item_id"] for item in ocr.json()["items"]] == ["a", "b"]
    assert ocr.json()["model"] == "ocr/model"
    assert ocr.json()["items"][0]["regions"] == [
        {
            "text": "text 10",
            "confidence": None,
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 1.0,
            "y_max": 1.0,
        }
    ]
    assert ocr.json()["items"][0]["raw_output"] is None
    assert ocr.json()["items"][1]["raw_output"] == [1, 2]

    embedded = request(
        app,
        "POST",
        "/v1/embeddings/dino",
        data={"item_ids": json.dumps(["a", "b"])},
        files=files,
    )
    assert embedded.status_code == 200
    assert embedded.json()["item_ids"] == ["a", "b"]
    assert embedded.json()["revision"] == "dino-sha"

    frames = np.zeros((3, 27, 48, 3), dtype=np.uint8)
    scored = request(
        app,
        "POST",
        "/v1/preprocessing/shot-scores",
        data={"request_id": "shot-1"},
        files=[("tensor", ("frames.npy", _npy(frames), "application/x-npy"))],
    )
    assert scored.status_code == 200
    assert scored.json()["request_id"] == "shot-1"
    assert len(scored.json()["scores"]) == 3

    audio = AudioReferenceRequest(
        request_id="asr-1",
        video_id="video-1",
        audio_url="https://s3.test/audio.flac?signature=test",
        audio_sha256="a" * 64,
    )
    transcript = request(
        app, "POST", "/v1/transcripts/asr", json=audio.model_dump(mode="json")
    )
    assert transcript.status_code == 200
    assert transcript.json()["segments"][0]["video_id"] == "video-1"

    diarization = DiarizationRequest(
        **audio.model_dump(), segments=[_segment()]
    )
    diarized = request(
        app,
        "POST",
        "/v1/transcripts/diarization",
        json=diarization.model_dump(mode="json"),
    )
    assert diarized.status_code == 200
    assert diarized.json()["segments"][0]["speaker_id"] == "SPEAKER_00"


def test_offline_inference_client_validates_returned_identity():
    def handler(request):
        if request.url.path == "/v1/embeddings/images":
            return httpx.Response(200, json={
                "model": "visual/model",
                "revision": "visual-sha",
                "dimension": 2,
                "normalized": True,
                "item_ids": ["frame-1"],
                "embeddings": [[0.0, 1.0]],
                "latency_ms": 1,
            })
        if request.url.path == "/v1/preprocessing/shot-scores":
            return httpx.Response(200, json={
                "request_id": "shot-1",
                "model": "transnet",
                "revision": "transnet-sha",
                "scores": [0.1, 0.2],
                "latency_ms": 1,
            })
        return httpx.Response(200, json={
            "request_id": "asr-1",
            "video_id": "video-1",
            "model": "asr/model",
            "revision": "asr-sha",
            "segments": [_segment().model_dump(mode="json")],
            "latency_ms": 1,
        })

    http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://model.test"
    )
    client = InferenceClient("https://model.test", client=http)
    embedded = client.embed_images(
        [Image.new("RGB", (2, 2))], item_ids=["frame-1"]
    )
    assert embedded.item_ids == ["frame-1"]

    scores = client.boundary_scores(
        np.zeros((2, 27, 48, 3), dtype=np.uint8),
        request_id="shot-1",
        source="shot",
    )
    assert scores.scores == [0.1, 0.2]

    transcript = client.transcribe_audio_reference(AudioReferenceRequest(
        request_id="asr-1",
        video_id="video-1",
        audio_url="https://s3.test/audio.flac?signature=test",
        audio_sha256="a" * 64,
    ))
    assert transcript.segments == [_segment()]


def test_runtime_does_not_construct_or_require_disabled_models():
    runtime = LLMService(LocalAdapter(
        LLMServiceConfig(),
        enable_caption=False,
        enable_visual_embedding=False,
        enable_caption_embedding=False,
        enable_reranker=False,
        enable_vqa=False,
    ))

    runtime.load()
    readiness = runtime.readiness()

    assert readiness.ready is True
    assert all(not status.enabled for status in readiness.models.values())
    assert all(not status.loaded for status in readiness.models.values())
    with pytest.raises(RuntimeError, match="embedding model is disabled"):
        runtime.embed_text(["query"])
    with pytest.raises(RuntimeError, match="caption model is disabled"):
        runtime.caption([Image.new("RGB", (1, 1))])
    with pytest.raises(RuntimeError, match="ocr model is disabled"):
        runtime.ocr([Image.new("RGB", (1, 1))])


def test_asr_readiness_requires_the_enabled_model_to_be_loaded():
    adapter = LocalAdapter(
        LLMServiceConfig(),
        enable_caption=False,
        enable_visual_embedding=False,
        enable_caption_embedding=False,
        enable_reranker=False,
        enable_vqa=False,
        enable_asr=True,
    )

    assert adapter.readiness().ready is False
    adapter.asr = object()
    assert adapter.readiness().ready is True


def test_asr_only_environment_does_not_construct_unrequested_models(
    monkeypatch,
):
    capability_flags = (
        "HCMAI_ENABLE_CAPTION",
        "HCMAI_ENABLE_OCR",
        "HCMAI_ENABLE_VISUAL_EMBEDDING",
        "HCMAI_ENABLE_CAPTION_EMBEDDING",
        "HCMAI_ENABLE_RERANKER",
        "HCMAI_ENABLE_VQA",
        "HCMAI_ENABLE_DIARIZATION",
    )
    for name in capability_flags:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HCMAI_ENABLE_ASR", "true")

    adapter = LocalAdapter.from_environment()

    assert adapter.enable_asr is True
    assert adapter.captioner is None
    assert adapter.ocr_adapter is None
    assert adapter.visual_encoder is None
    assert adapter.caption_encoder is None
    assert adapter.reranker is None
    assert adapter.vqa_model is None
    assert adapter.enable_diarization is False


def test_caption_and_ocr_share_one_identically_pinned_florence_backend():
    config = LLMServiceConfig.from_yaml("thundercompute/config.yaml")
    model = object()
    processor = object()
    captioner = SimpleNamespace(
        config=config.caption_generation,
        model=model,
        processor=processor,
        resolved_revision=config.caption_generation.revision,
        resolve_revision=lambda: config.caption_generation.revision,
    )
    ocr = FlorenceAdapter(
        OCRConfig(
            checkpoint=config.caption_generation.model_checkpoint,
            revision=config.caption_generation.revision,
            device=config.caption_generation.device,
            dtype=config.caption_generation.dtype,
        )
    )
    adapter = LocalAdapter(
        config,
        captioner=captioner,
        ocr_adapter=ocr,
        enable_caption=True,
        enable_visual_embedding=False,
        enable_caption_embedding=False,
        enable_reranker=False,
        enable_vqa=False,
        enable_ocr=True,
    )

    adapter.load()

    assert ocr.model is model
    assert ocr.processor is processor
