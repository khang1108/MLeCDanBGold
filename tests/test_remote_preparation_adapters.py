from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from hcmai.common.config import ASRConfig, DiarizationConfig, EncoderConfig
from hcmai.common.schemas import (
    AudioReferenceRequest,
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
            segments=[
                payload.segments[0].model_copy(
                    update={"speaker_id": "SPEAKER_00"}
                )
            ],
            latency_ms=1,
        )


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


def test_remote_enrichment_and_embedding_adapters_validate_pins() -> None:
    client = FakeClient()
    assert not hasattr(client.readiness().capabilities, "multi_image_vqa")
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


def test_remote_image_embedding_uses_configured_batch_ceiling() -> None:
    """A large visual configuration is not reduced to the legacy API cap."""

    class RecordingClient:
        """Return aligned unit vectors while retaining request batch sizes."""

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def embed_images(self, images, *, source="visual", item_ids=None):
            self.batch_sizes.append(len(images))
            return EmbeddingResponse(
                model="visual",
                revision=SHA,
                dimension=2,
                normalized=True,
                item_ids=item_ids,
                embeddings=[[0.0, 1.0]] * len(images),
                latency_ms=1,
            )

    client = RecordingClient()
    adapter = RemoteImageEmbeddingAdapter(
        client,
        EncoderConfig(model_name="visual", revision=SHA, batch_size=128),
    )

    vectors = adapter.encode_images([Image.new("RGB", (2, 2))] * 130)

    assert vectors.shape == (130, 2)
    assert client.batch_sizes == [128, 2]


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
    assert segments[0].model_name == "asr"
    assert segments[0].model_revision == SHA
    assert segments[0].artifact_version == "asr-segment-v1"

    diarized = diarization.assign_speakers(video, segments)
    assert diarized == [
        segments[0].model_copy(update={"speaker_id": "SPEAKER_00"})
    ]
    assert diarized[0].speaker_id == "SPEAKER_00"
    assert diarized[0].model_name == "asr"
    assert diarized[0].model_revision == SHA
    assert diarized[0].artifact_version == "asr-segment-v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "other/asr"),
        ("model_revision", "b" * 40),
        ("artifact_version", "other-segment-v1"),
    ],
)
def test_remote_asr_rejects_segment_lineage_conflicting_with_envelope(
    field: str,
    value: str,
) -> None:
    """Do not accept provider segments that contradict a trusted ASR envelope."""

    client, references = FakeClient(), FakeReferences()
    client.transcribe_audio_reference = lambda payload: TranscriptInferenceResponse(
        request_id=payload.request_id,
        video_id=payload.video_id,
        model="asr",
        revision=SHA,
        segments=[_segment().model_copy(update={field: value})],
        latency_ms=1,
    )
    adapter = RemoteASRAdapter(
        client, ASRConfig(model_name="asr", revision=SHA), references
    )

    with pytest.raises(ValueError, match=f"segment {field} conflicts"):
        adapter.transcribe(Path("L21_V001.mp4"), "L21_V001")


@pytest.mark.parametrize(
    "update",
    [
        {"text": "mutated transcript"},
        {"start_ms": 1},
        {"end_ms": 999},
        {"language": "en"},
    ],
)
def test_remote_diarization_rejects_canonical_asr_mutation(
    update: dict[str, object],
) -> None:
    """Accept only speaker labels from the remote diarization provider."""

    client, references = FakeClient(), FakeReferences()
    asr = RemoteASRAdapter(
        client, ASRConfig(model_name="asr", revision=SHA), references
    )
    diarization = RemoteDiarizationAdapter(
        client,
        DiarizationConfig(model_name="diarization", revision=SHA),
        references,
    )
    segments = asr.transcribe(Path("L21_V001.mp4"), "L21_V001")
    client.diarize_audio_reference = lambda payload: TranscriptInferenceResponse(
        request_id=payload.request_id,
        video_id=payload.video_id,
        model="diarization",
        revision=SHA,
        segments=[payload.segments[0].model_copy(update=update)],
        latency_ms=1,
    )

    with pytest.raises(ValueError, match="changed canonical ASR segment"):
        diarization.assign_speakers(Path("L21_V001.mp4"), segments)
