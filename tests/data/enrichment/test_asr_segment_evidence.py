"""Verify ASR remains versioned timeline evidence with optional frame alignment."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from hcmai.common.config import ASRConfig
from hcmai.common.schemas import ProcessingStatus, TranscriptSegment
from hcmai.data.enrichment.transcripts.adapters.asr import ASRAdapter, DecodedAudio
from hcmai.data.enrichment.transcripts.manifest import (
    SourceFingerprint,
    TranscriptManifest,
)
from hcmai.data.enrichment.transcripts.store import TranscriptStore


def _legacy_segment() -> dict[str, object]:
    """Return the segment shape persisted before explicit ASR lineage."""

    return {
        "segment_id": "v1_segment_000000",
        "video_id": "v1",
        "segment_index": 0,
        "start_ms": 1_000,
        "end_ms": 2_000,
        "text": "hello",
        "language": "en",
        "speaker_id": None,
    }


def test_legacy_segment_defaults_and_round_trips_through_store(
    tmp_path: Path,
) -> None:
    """Load old rows while exposing explicit default lineage and status."""

    legacy = TranscriptSegment.model_validate(_legacy_segment())
    assert legacy.confidence is None
    assert legacy.status is ProcessingStatus.COMPLETED
    assert legacy.model_name is None
    assert legacy.model_revision is None
    assert legacy.artifact_version == "asr-segment-v1"

    path = tmp_path / "segments.parquet"
    pd.DataFrame([legacy.model_dump(mode="json")]).to_parquet(path, index=False)

    loaded = TranscriptStore(path).get(legacy.segment_id)
    assert loaded == legacy
    assert TranscriptSegment.model_validate_json(loaded.model_dump_json()) == legacy


def test_transcript_segment_validates_duration_and_confidence() -> None:
    """Keep positive media duration and calibrated confidence bounds strict."""

    with pytest.raises(ValidationError, match="end_ms must be greater"):
        TranscriptSegment.model_validate(_legacy_segment() | {"end_ms": 1_000})
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        TranscriptSegment.model_validate(_legacy_segment() | {"confidence": 1.01})


def test_local_asr_stamps_lineage_without_inventing_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stamp configured and resolved model identity on generated segments."""

    config = ASRConfig(
        device="cpu",
        model_name="test/asr",
        revision="a" * 40,
    )
    model = type("Model", (), {
        "config": type("Config", (), {"_commit_hash": "b" * 40})(),
    })()
    adapter = ASRAdapter(config, model=model, processor=object(), vad_model=object())
    monkeypatch.setattr(
        adapter,
        "_speech_regions",
        lambda _waveform: [{"start": 0, "end": 16_000}],
    )
    monkeypatch.setattr(
        adapter,
        "_infer_batch",
        lambda _clips: [{"transcription": "hello", "language": "English"}],
    )

    rows = adapter.transcribe_audio(
        DecodedAudio(np.zeros(16_000, dtype=np.float32), 16_000, 250),
        "v1",
    )

    assert len(rows) == 1
    assert rows[0].model_name == "test/asr"
    assert rows[0].model_revision == "b" * 40
    assert rows[0].artifact_version == "asr-segment-v1"
    assert rows[0].confidence is None
    assert (rows[0].start_ms, rows[0].end_ms, rows[0].language) == (
        250,
        1_250,
        "english",
    )


def test_manifest_marks_segment_source_and_compatibility_alignment() -> None:
    """Make the source-of-truth and context boundary machine-readable."""

    manifest = TranscriptManifest(
        video_id="v1",
        source=SourceFingerprint(size_bytes=1, sha256="a" * 64),
        config_sha256="b" * 64,
        asr_model="test/asr",
        asr_revision="c" * 40,
        diarization_enabled=False,
        schema_version="transcript-segment-v1",
        pipeline_version="transcript-pipeline-v1",
        segment_count=1,
        status="completed",
    )

    assert manifest.source_of_truth == "transcript segments"
    assert manifest.frame_alignment == "derived compatibility view"
    assert manifest.context_dependency == "none"


def test_frame_context_modules_do_not_depend_on_asr_compatibility_view() -> None:
    """Prevent deterministic FrameContext from acquiring frame-aligned ASR inputs."""

    source_root = Path(__file__).resolve().parents[3] / "src" / "hcmai"
    forbidden = {
        "materialize_asr_enrichment",
        "ASRStore",
    }

    context_imports: set[str] = set()
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_parts = path.relative_to(source_root).parts
        owns_context = any("context" in part.casefold() for part in relative_parts)
        owns_frame_context = any(
            isinstance(node, ast.ClassDef) and node.name == "FrameContext"
            for node in tree.body
        )
        if not owns_context and not owns_frame_context:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                context_imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                context_imports.update(
                    alias.name.rsplit(".", maxsplit=1)[-1]
                    for alias in node.names
                )

    transcript_imports: set[str] = set()
    transcript_paths = [source_root / "common/schemas/transcript.py"]
    transcript_paths.extend(
        sorted((source_root / "data/enrichment/transcripts").rglob("*.py"))
    )
    for path in transcript_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                transcript_imports.update(alias.name for alias in node.names)

    assert context_imports.isdisjoint(forbidden)
    assert "FrameContext" not in transcript_imports
