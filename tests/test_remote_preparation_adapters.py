from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hcmai.common.config import ASRConfig, DiarizationConfig, EncoderConfig
from hcmai.common.schemas import (
    AudioReferenceRequest,
    BoundaryScoreResponse,
    EmbeddingResponse,
    InferenceReadiness,
    ModelStatus,
    OCRItem,
    OCRResponse,
    TranscriptInferenceResponse,
    TranscriptSegment,
)
from hcmai.data.enrichment.ocr.adapters.remote import RemoteOCRAdapter
from hcmai.data.enrichment.ocr.config import OCRConfig
from hcmai.data.enrichment.transcripts.adapters.remote import (
    RemoteASRAdapter,
    RemoteDiarizationAdapter,
)
from hcmai.data.preprocessing.adapters.remote import (
    RemoteDinoEncoder,
    RemoteEfficientGEBDDetector,
    RemoteTransNetDetector,
)
from hcmai.data.preprocessing.video import FrameMeta
from hcmai.retrieval.embedding.adapters.remote import (
    RemoteEmbeddingAdapter,
    RemoteImageEmbeddingAdapter,
)

SHA = "a" * 40


def _segment(speaker_id: str | None = None) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id="L21_V001_segment_000000",
        video_id="L21_V001",
        segment_index=0,
        start_ms=0,
        end_ms=1000,
        text="xin chao",
        language="vi",
        speaker_id=speaker_id,
    )


class FakeClient:
    def boundary_scores(self, frames, *, request_id, source):
        model = "transnet" if source == "shot" else "gebd"
        return BoundaryScoreResponse(
            request_id=request_id,
            model=model,
            revision=SHA,
            scores=np.linspace(0, 1, len(frames)).tolist(),
            latency_ms=1,
        )

    def embed_images(self, images, *, source="visual", item_ids=None):
        model = "dino" if source == "dino" else "visual"
        return EmbeddingResponse(
            model=model,
            revision=SHA,
            dimension=2,
            normalized=True,
            item_ids=item_ids,
            embeddings=[[0.0, 1.0]] * len(images),
            latency_ms=1,
        )

    def ocr(self, images):
        return OCRResponse(
            model="ocr",
            revision=SHA,
            items=[OCRItem(item_id=str(i), text=f"text-{i}") for i in range(len(images))],
            latency_ms=1,
        )

    def readiness(self):
        return InferenceReadiness(
            ready=True,
            models={
                name: ModelStatus(loaded=True, checkpoint=model, revision=SHA)
                for name, model in {
                    "ocr": "ocr",
                    "asr": "asr",
                    "diarization": "diarization",
                }.items()
            },
        )

    def transcribe_audio_reference(self, payload):
        return TranscriptInferenceResponse(
            request_id=payload.request_id,
            video_id=payload.video_id,
            model="asr",
            revision=SHA,
            segments=[_segment()],
            latency_ms=1,
        )

    def diarize_audio_reference(self, payload):
        return TranscriptInferenceResponse(
            request_id=payload.request_id,
            video_id=payload.video_id,
            model="diarization",
            revision=SHA,
            segments=[_segment("SPEAKER_00")],
            latency_ms=1,
        )


class FakeSource:
    def to_image(self):
        return Image.new("RGB", (8, 8), "red")


class FakeReferences:
    def reference(self, video_path: Path, video_id: str, sample_rate: int):
        assert video_path.name == "L21_V001.mp4"
        assert sample_rate == 16_000
        return AudioReferenceRequest(
            request_id="audio-1",
            video_id=video_id,
            audio_url="https://s3.test/audio.flac?signature=test",
            audio_sha256="b" * 64,
            sample_rate=sample_rate,
        )


def test_remote_preprocessing_adapters_preserve_order_and_alignment() -> None:
    client = FakeClient()
    frames = np.zeros((3, 27, 48, 3), dtype=np.uint8)
    transnet = RemoteTransNetDetector(
        client, model_name="transnet", revision=SHA
    )
    assert transnet.score(Path("video.mp4"), frames).shape == (3,)

    dino = RemoteDinoEncoder(client, model_name="dino", revision=SHA)
    assert dino.encode([Image.new("RGB", (2, 2))] * 2).shape == (2, 2)

    gebd = RemoteEfficientGEBDDetector(
        client,
        model_name="gebd",
        revision=SHA,
        sequence_length=2,
        overlap=1,
        resolution=8,
    )
    for index in range(3):
        gebd.update(
            FrameMeta(
                video_id="L21_V001",
                decode_index=index,
                frame_idx=index,
                pts=index,
                time_base="1/10",
                timestamp_ms=index * 100,
                width=8,
                height=8,
            ),
            FakeSource(),
        )
    assert gebd.scores(3).shape == (3,)


def test_remote_enrichment_and_embedding_adapters_validate_pins() -> None:
    client = FakeClient()
    ocr = RemoteOCRAdapter(
        client, OCRConfig(backend="remote", checkpoint="ocr", revision=SHA)
    )
    assert ocr.resolve_revision() == SHA
    assert ocr.recognize_batch([Image.new("RGB", (2, 2))])[0].text == "text-0"

    config = EncoderConfig(model_name="visual", revision=SHA, batch_size=1)
    encoder = RemoteImageEmbeddingAdapter(client, config)
    vectors = encoder.encode_images([Image.new("RGB", (2, 2))] * 2)
    assert vectors.shape == (2, 2)

    text = RemoteEmbeddingAdapter(client, config, embedding_dim=2)
    client.embed_text = lambda values, source="visual": EmbeddingResponse(
        model="visual",
        revision=SHA,
        dimension=2,
        normalized=True,
        embeddings=[[0.0, 1.0]] * len(values),
        latency_ms=1,
    )
    assert text.encode_text(["xin chao"]).shape == (1, 2)


def test_remote_embedding_rejects_false_normalization_claim() -> None:
    client = FakeClient()
    client.embed_images = lambda images, **kwargs: EmbeddingResponse(
        model="visual",
        revision=SHA,
        dimension=2,
        normalized=True,
        item_ids=kwargs["item_ids"],
        embeddings=[[1.0, 1.0]] * len(images),
        latency_ms=1,
    )
    adapter = RemoteImageEmbeddingAdapter(
        client, EncoderConfig(model_name="visual", revision=SHA)
    )
    with pytest.raises(ValueError, match="L2-normalized"):
        adapter.encode_images([Image.new("RGB", (2, 2))])


def test_remote_transcript_adapters_share_audio_reference_and_keep_segments() -> None:
    client, references = FakeClient(), FakeReferences()
    asr = RemoteASRAdapter(
        client, ASRConfig(model_name="asr", revision=SHA), references
    )
    diarization = RemoteDiarizationAdapter(
        client,
        DiarizationConfig(model_name="diarization", revision=SHA),
        references,
    )
    video = Path("L21_V001.mp4")
    segments = asr.transcribe(video, "L21_V001")
    assert segments == [_segment()]
    assert diarization.assign_speakers(video, segments) == [_segment("SPEAKER_00")]
