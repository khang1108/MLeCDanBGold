from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from hcmai.common.config import ASRConfig
from hcmai.common.schemas import (
    FrameEnrichment,
    FrameRecord,
    ProcessingStatus,
    RetrievalSource,
    TranscriptSegment,
)
from offline.enrichment.transcripts.adapters.asr import (
    ASRAdapter,
    _validate_segments,
    read_audio,
)
from offline.enrichment.transcripts.adapters.diarization import _speaker_id
from offline.enrichment.transcripts.manifest import (
    SourceFingerprint,
    TranscriptManifest,
    reusable_transcript,
)
from offline.enrichment.transcripts.materialize import (
    materialize_asr_enrichment,
    materialize_transcript_artifact,
    write_asr_enrichment,
)
from offline.enrichment.transcripts.prepare import prepare_transcripts
from offline.enrichment.transcripts.publication import publish_staged
from hcmai.corpus.stores import ASRStore, FrameStore


class _Frame:
    def __init__(self, pts, time_base):
        self.pts = pts
        self.time_base = time_base


class _Resampled:
    @staticmethod
    def to_ndarray():
        return np.asarray([[0.25, -0.25]], dtype=np.float32)


class _Resampler:
    def __init__(self, **_kwargs):
        pass

    def resample(self, frame):
        return [] if frame is None else [_Resampled()]


class _Container:
    def __init__(self, frame):
        self.frame = frame
        self.stream = SimpleNamespace(
            start_time=3,
            time_base=Fraction(1, 2),
        )
        self.streams = SimpleNamespace(audio=[self.stream])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def decode(self, _stream):
        return [self.frame]


@pytest.mark.parametrize(
    ("pts", "time_base", "expected"),
    [
        (10, Fraction(1, 100), 100),
        (None, None, 1_500),
    ],
)
def test_audio_start_uses_first_pts_then_stream_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pts,
    time_base,
    expected: int,
) -> None:
    fake_av = SimpleNamespace(
        open=lambda _path: _Container(_Frame(pts, time_base)),
        AudioResampler=_Resampler,
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)

    decoded = read_audio(tmp_path / "video.mp4", 16_000)

    assert decoded.start_ms == expected
    assert decoded.sample_rate == 16_000
    assert decoded.samples.dtype == np.float32
    assert decoded.samples.flags.writeable is False


def test_diarization_turns_are_translated_to_media_time() -> None:
    segment = TranscriptSegment(
        segment_id="v1_segment_000000",
        video_id="v1",
        segment_index=0,
        start_ms=5_100,
        end_ms=5_900,
        text="hello",
        language="en",
    )
    turn = SimpleNamespace(start=0.0, end=1.0)

    assert _speaker_id(
        segment, [(turn, "SPEAKER_00")], audio_start_ms=5_000
    ) == "SPEAKER_00"


def _manifest() -> TranscriptManifest:
    return TranscriptManifest(
        video_id="v1",
        source=SourceFingerprint(size_bytes=3, sha256="a" * 64),
        config_sha256="b" * 64,
        asr_model="asr/model",
        asr_revision="c" * 40,
        diarization_enabled=True,
        diarization_model="diarizer/model",
        diarization_revision="d" * 40,
        schema_version="schema-v1",
        pipeline_version="pipeline-v1",
        segment_count=0,
        status="completed",
    )


@pytest.mark.parametrize(
    "update",
    [
        {"source": SourceFingerprint(size_bytes=4, sha256="e" * 64)},
        {"video_id": "v2"},
        {"config_sha256": "e" * 64},
        {"asr_model": "other/asr"},
        {"asr_revision": "e" * 40},
        {"diarization_model": "other/diarizer"},
        {"diarization_revision": "e" * 40},
        {
            "diarization_enabled": False,
            "diarization_model": None,
            "diarization_revision": None,
        },
        {"schema_version": "schema-v2"},
        {"pipeline_version": "pipeline-v2"},
    ],
)
def test_every_manifest_identity_mismatch_invalidates_resume(
    tmp_path: Path, update: dict[str, object]
) -> None:
    output = tmp_path / "v1.parquet"
    output.write_bytes(b"present")
    manifest_path = tmp_path / "v1.manifest.json"
    manifest_path.write_text(
        _manifest().model_copy(update=update).model_dump_json(),
        encoding="utf-8",
    )

    assert not reusable_transcript(output, manifest_path, _manifest(), [])


class _ASR:
    def __init__(
        self,
        error: Exception | None = None,
        no_speech_video_ids: set[str] | None = None,
    ):
        self.config = ASRConfig(device="cpu")
        self.resolved_revision = self.config.revision
        self.error = error
        self.no_speech_video_ids = no_speech_video_ids or set()
        self.calls = 0

    def transcribe(self, _path, video_id):
        self.calls += 1
        if self.error:
            raise self.error
        if video_id in self.no_speech_video_ids:
            return []
        return [TranscriptSegment(
            segment_id=f"{video_id}_segment_000000",
            video_id=video_id,
            segment_index=0,
            start_ms=1_000,
            end_ms=2_000,
            text="hello",
            language="en",
        )]


def test_optional_diarization_preserves_none_and_source_change_reprocesses(
    tmp_path: Path,
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    source = videos / "v1.mp4"
    source.write_bytes(b"one")
    output = tmp_path / "transcripts"
    engine = _ASR()

    first = prepare_transcripts(videos, output, cast(ASRAdapter, engine))
    second = prepare_transcripts(videos, output, cast(ASRAdapter, engine))
    source.write_bytes(b"two-changed")
    third = prepare_transcripts(videos, output, cast(ASRAdapter, engine))

    assert not first.failed and not second.failed and not third.failed
    assert engine.calls == 2
    table = pd.read_parquet(output / "v1/v1.parquet")
    assert pd.isna(table.iloc[0]["speaker_id"])
    manifest = json.loads((output / "v1/v1.manifest.json").read_text())
    assert manifest["diarization_enabled"] is False


def test_non_monotonic_final_intervals_are_rejected() -> None:
    records = [
        _segment(0, "v1", 2_000, 3_000, "later"),
        _segment(1, "v1", 1_000, 2_500, "earlier"),
    ]
    with pytest.raises(ValueError, match="monotonic"):
        _validate_segments(records)


def test_failed_video_records_only_safe_category(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "v1.mp4").write_bytes(b"video")
    output = tmp_path / "transcripts"
    engine = _ASR(RuntimeError("provider token=super-secret"))

    report = prepare_transcripts(videos, output, cast(ASRAdapter, engine))

    assert report.failed == {"v1": "RuntimeError"}
    failure = (output / "v1/v1.failure.json").read_text(encoding="utf-8")
    assert "RuntimeError" in failure
    assert "super-secret" not in failure


def test_duplicate_video_id_with_same_size_different_content_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    for group, content in (("Videos_a", b"one"), ("Videos_b", b"two")):
        directory = root / group / "video"
        directory.mkdir(parents=True)
        (directory / "v1.mp4").write_bytes(content)

    with pytest.raises(ValueError, match="different source content"):
        prepare_transcripts(root, tmp_path / "output", cast(ASRAdapter, _ASR()))


def _frame(frame_id: str, video_id: str, timestamp_ms: int) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        video_id=video_id,
        frame_idx=timestamp_ms,
        timestamp_ms=timestamp_ms,
        image_path=f"{frame_id}.jpg",
        width=8,
        height=6,
    )


def _segment(index: int, video_id: str, start: int, end: int, text: str):
    return TranscriptSegment(
        segment_id=f"{video_id}_segment_{index:06d}",
        video_id=video_id,
        segment_index=index,
        start_ms=start,
        end_ms=end,
        text=text,
        language="en",
    )


def test_materialization_uses_half_open_overlap_dedup_and_evaluated_videos() -> None:
    frames = [
        _frame("f-boundary", "v1", 1_500),
        _frame("f-window", "v1", 1_750),
        _frame("f-no-speech", "v2", 1_000),
        _frame("f-unknown", "v3", 1_000),
    ]
    segments = [
        _segment(0, "v1", 1_000, 1_500, "Before"),
        _segment(1, "v1", 1_500, 2_000, "At"),
        _segment(2, "v1", 1_600, 1_900, " at "),
    ]

    boundary = materialize_asr_enrichment(
        frames[:1],
        segments,
        evaluated_video_ids={"v1"},
        window_ms=0,
        enrichment_version="asr-v1",
        model_name="asr@revision:pipeline-v1",
    )
    rows = materialize_asr_enrichment(
        frames,
        segments,
        evaluated_video_ids={"v1", "v2"},
        window_ms=300,
        enrichment_version="asr-v1",
        model_name="asr@revision:pipeline-v1",
    )

    assert boundary[0].asr_text == "At"
    assert [row.frame_id for row in rows] == [
        "f-boundary", "f-window", "f-no-speech"
    ]
    assert rows[1].asr_text == "Before At"
    assert rows[2].asr_text is None
    assert rows[2].status is ProcessingStatus.COMPLETED


def test_materialization_rejects_unknown_video_identity() -> None:
    with pytest.raises(ValueError, match="unknown canonical videos"):
        materialize_asr_enrichment(
            [_frame("f1", "v1", 100)],
            [_segment(0, "foreign", 0, 200, "text")],
            evaluated_video_ids={"v1", "foreign"},
            window_ms=10,
            enrichment_version="asr-v1",
            model_name="asr@revision:pipeline-v1",
        )


def test_materialization_rejects_unknown_frame_foreign_key(tmp_path: Path) -> None:
    row = FrameEnrichment(
        frame_id="invented",
        asr_text="text",
        enrichment_version="asr-v1",
        model_name="asr@revision:pipeline-v1",
    )
    with pytest.raises(ValueError, match="unknown frame IDs"):
        write_asr_enrichment(
            tmp_path / "asr.parquet", [row], canonical_frame_ids={"f1"}
        )


def test_materialized_rows_validate_with_offline_stores(tmp_path: Path) -> None:
    frame = _frame("f1", "v1", 1_500)
    frames_path = tmp_path / "frames.parquet"
    pd.DataFrame([frame.model_dump(mode="json")]).to_parquet(frames_path, index=False)
    output = tmp_path / "asr.parquet"
    rows = materialize_asr_enrichment(
        [frame],
        [_segment(0, "v1", 1_000, 2_000, "hello")],
        evaluated_video_ids={"v1"},
        window_ms=0,
        enrichment_version="asr-v1",
        model_name="asr@revision:pipeline-v1",
    )

    write_asr_enrichment(output, rows, canonical_frame_ids={"f1"})
    FrameStore(frames_path)
    evidence = ASRStore(output)

    assert all(FrameEnrichment.model_validate(row.model_dump()) for row in rows)
    assert evidence.get_text("f1") == "hello"


def test_completed_manifests_materialize_speech_and_no_speech_videos(
    tmp_path: Path,
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    for video_id in ("v1", "v2"):
        (videos / f"{video_id}.mp4").write_bytes(video_id.encode())
    transcripts = tmp_path / "transcripts"
    prepare_transcripts(
        videos,
        transcripts,
        cast(ASRAdapter, _ASR(no_speech_video_ids={"v2"})),
    )
    frames = [_frame("f1", "v1", 1_500), _frame("f2", "v2", 1_500)]
    frames_path = tmp_path / "frames.parquet"
    pd.DataFrame(
        [frame.model_dump(mode="json") for frame in frames]
    ).to_parquet(frames_path, index=False)
    output = tmp_path / "asr.parquet"

    materialize_transcript_artifact(
        frames_path,
        transcripts,
        output,
        window_ms=0,
        enrichment_version="asr-v1",
        model_name="asr@revision:pipeline-v1",
    )
    FrameStore(frames_path)
    evidence = ASRStore(output)

    assert evidence.get_text("f1") == "hello"
    assert evidence.get_text("f2") is None
    assert [row.frame_id for row in evidence.iter_records()] == [
        "f1", "f2"
    ]


def test_materialization_validation_failure_preserves_previous_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import offline.enrichment.transcripts.materialize as module

    target = tmp_path / "asr.parquet"
    target.write_bytes(b"old-valid-artifact")
    row = FrameEnrichment(
        frame_id="f1",
        asr_text="hello",
        enrichment_version="asr-v1",
        model_name="model@revision",
    )
    monkeypatch.setattr(
        module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid staging")),
    )

    with pytest.raises(ValueError, match="invalid staging"):
        write_asr_enrichment(target, [row], canonical_frame_ids={"f1"})
    assert target.read_bytes() == b"old-valid-artifact"


def test_pair_publication_failure_restores_both_previous_files(tmp_path: Path) -> None:
    first, second = tmp_path / "data.parquet", tmp_path / "manifest.json"
    first.write_bytes(b"old-data")
    second.write_bytes(b"old-manifest")
    staged_first = tmp_path / ".data.staging"
    staged_second = tmp_path / ".manifest.staging"
    staged_first.write_bytes(b"new-data")
    staged_second.write_bytes(b"new-manifest")

    def fail_second(source, target):
        if Path(source) == staged_second:
            raise OSError("promotion failed")
        os.replace(source, target)

    with pytest.raises(OSError, match="promotion failed"):
        publish_staged(
            {first: staged_first, second: staged_second}, replace=fail_second
        )

    assert first.read_bytes() == b"old-data"
    assert second.read_bytes() == b"old-manifest"
