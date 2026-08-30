"""Tests for reusable local ASR transcript/vector lineage validation.

Covers valid reuse, missing archive-video coverage, missing manifest,
duplicate/invalid segments, index/vector mismatches, corrupt checksums,
unrelated source videos, and deterministic fingerprints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcmai.common.utils.io import write_json
from offline.ingestion.custom_pipeline.asr import (
    require_asr_video_coverage,
    validate_asr_source,
)
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

pytest.importorskip("faiss")


def _manifest_dict(video_id: str, segment_count: int) -> dict[str, object]:
    return {
        "video_id": video_id,
        "source": {"size_bytes": 1024, "sha256": "a" * 64},
        "config_sha256": "b" * 64,
        "asr_model": "Qwen/Qwen3-ASR-1.7B-hf",
        "asr_revision": "c" * 40,
        "diarization_enabled": False,
        "diarization_model": None,
        "diarization_revision": None,
        "schema_version": "transcript-segment-v1",
        "pipeline_version": "transcript-pipeline-v1",
        "segment_count": segment_count,
        "status": "completed",
        "failure_category": None,
    }


def _write_transcript(transcripts_root: Path, video_id: str, segments: list[dict[str, object]]) -> None:
    video_dir = transcripts_root / video_id.split("_", 1)[0]
    video_dir.mkdir(parents=True, exist_ok=True)
    write_json(_manifest_dict(video_id, len(segments)), video_dir / f"{video_id}.manifest.json")
    table = pd.DataFrame(
        [
            {
                "segment_id": segment["segment_id"],
                "video_id": video_id,
                "start_ms": segment["start_ms"],
                "end_ms": segment["end_ms"],
            }
            for segment in segments
        ],
        columns=["segment_id", "video_id", "start_ms", "end_ms"],
    )
    table.to_parquet(video_dir / f"{video_id}.parquet")


def _build_index(index_root: Path, rows: list[dict[str, object]]) -> None:
    mapping = pd.DataFrame(
        [
            {
                "embedding_index": position,
                "segment_id": row["segment_id"],
                "video_id": row["video_id"],
                "segment_index": index,
                "start_ms": row["start_ms"],
                "end_ms": row["end_ms"],
            }
            for position, (index, row) in enumerate(
                (i, r) for i, r in enumerate(rows)
            )
        ]
    )
    vectors = np.eye(len(rows), dtype=np.float32)
    index = SegmentDenseIndex.build(vectors, mapping, dataset_version="v1", model_name="test-model")
    index.save(index_root)


def _default_fixture(tmp_path: Path) -> tuple[Path, Path]:
    transcripts_root = tmp_path / "transcripts"
    index_root = tmp_path / "index"

    v1_segments = [
        {"segment_id": "L01_V001-000", "start_ms": 0, "end_ms": 1000},
        {"segment_id": "L01_V001-001", "start_ms": 1000, "end_ms": 2000},
    ]
    v2_segments = [{"segment_id": "L01_V002-000", "start_ms": 0, "end_ms": 1500}]
    v9_segments = [{"segment_id": "L01_V009-000", "start_ms": 0, "end_ms": 500}]

    _write_transcript(transcripts_root, "L01_V001", v1_segments)
    _write_transcript(transcripts_root, "L01_V002", v2_segments)
    _write_transcript(transcripts_root, "L01_V009", v9_segments)

    rows = (
        [{"video_id": "L01_V001", **s} for s in v1_segments]
        + [{"video_id": "L01_V002", **s} for s in v2_segments]
        + [{"video_id": "L01_V009", **s} for s in v9_segments]
    )
    _build_index(index_root, rows)
    return transcripts_root, index_root


def test_valid_reuse_bundle_covers_requested_videos(tmp_path: Path) -> None:
    transcripts_root, index_root = _default_fixture(tmp_path)
    bundle = validate_asr_source(transcripts_root, index_root, ["L01_V001", "L01_V002"])
    assert bundle.video_ids == ("L01_V001", "L01_V002")
    assert bundle.segment_count == 3


def test_unrelated_source_videos_in_index_are_accepted(tmp_path: Path) -> None:
    # L01_V009 exists in transcripts/index but is not requested; it must not
    # cause a failure even though it is present in the same index bundle.
    transcripts_root, index_root = _default_fixture(tmp_path)
    bundle = validate_asr_source(transcripts_root, index_root, ["L01_V001"])
    assert bundle.video_ids == ("L01_V001",)


def test_require_asr_video_coverage_rejects_missing_archive_video(tmp_path: Path) -> None:
    transcripts_root, index_root = _default_fixture(tmp_path)
    bundle = validate_asr_source(transcripts_root, index_root, ["L01_V001", "L01_V002"])
    with pytest.raises(ValueError, match="missing archive video"):
        require_asr_video_coverage(bundle, ["L01_V001", "L01_V003"])


def test_require_asr_video_coverage_accepts_full_coverage(tmp_path: Path) -> None:
    transcripts_root, index_root = _default_fixture(tmp_path)
    bundle = validate_asr_source(transcripts_root, index_root, ["L01_V001", "L01_V002"])
    require_asr_video_coverage(bundle, ["L01_V001"])  # subset is fine


def test_missing_transcript_manifest_is_rejected(tmp_path: Path) -> None:
    transcripts_root, index_root = _default_fixture(tmp_path)
    with pytest.raises(ValueError, match="missing transcript manifest"):
        validate_asr_source(transcripts_root, index_root, ["L01_V999"])


def test_duplicate_segment_id_in_transcript_parquet_is_rejected(tmp_path: Path) -> None:
    transcripts_root = tmp_path / "transcripts"
    index_root = tmp_path / "index"
    segments = [
        {"segment_id": "dup", "start_ms": 0, "end_ms": 1000},
        {"segment_id": "dup", "start_ms": 1000, "end_ms": 2000},
    ]
    _write_transcript(transcripts_root, "L01_V001", segments)
    # The index itself must stay internally valid; only the transcript
    # parquet under test contains the duplicate segment_id.
    index_segments = [{"segment_id": "a", "start_ms": 0, "end_ms": 1000}]
    _build_index(index_root, [{"video_id": "L01_V001", **s} for s in index_segments])
    with pytest.raises(ValueError, match="duplicate segment_id"):
        validate_asr_source(transcripts_root, index_root, ["L01_V001"])


def test_invalid_segment_interval_is_rejected(tmp_path: Path) -> None:
    transcripts_root = tmp_path / "transcripts"
    index_root = tmp_path / "index"
    segments = [{"segment_id": "s1", "start_ms": 1000, "end_ms": 1000}]
    _write_transcript(transcripts_root, "L01_V001", segments)
    # The index itself must stay internally valid; only the transcript
    # parquet under test contains the zero-duration interval.
    index_segments = [{"segment_id": "s1", "start_ms": 0, "end_ms": 1000}]
    _build_index(index_root, [{"video_id": "L01_V001", **s} for s in index_segments])
    with pytest.raises(ValueError, match="invalid segment interval"):
        validate_asr_source(transcripts_root, index_root, ["L01_V001"])


def test_index_vector_count_mismatch_is_rejected(tmp_path: Path) -> None:
    transcripts_root = tmp_path / "transcripts"
    index_root = tmp_path / "index"
    # Manifest/parquet claim two segments, but only one is indexed.
    segments = [
        {"segment_id": "s1", "start_ms": 0, "end_ms": 1000},
        {"segment_id": "s2", "start_ms": 1000, "end_ms": 2000},
    ]
    _write_transcript(transcripts_root, "L01_V001", segments)
    _build_index(index_root, [{"video_id": "L01_V001", **segments[0]}])
    with pytest.raises(ValueError, match="disagrees with transcript segment count"):
        validate_asr_source(transcripts_root, index_root, ["L01_V001"])


def test_silent_video_with_an_empty_transcript_is_accepted(tmp_path: Path) -> None:
    transcripts_root, index_root = _default_fixture(tmp_path)
    _write_transcript(transcripts_root, "L01_V003", [])

    bundle = validate_asr_source(transcripts_root, index_root, ["L01_V001", "L01_V003"])

    assert bundle.video_ids == ("L01_V001", "L01_V003")
    assert bundle.segment_count == 2


def test_corrupt_index_checksum_is_rejected(tmp_path: Path) -> None:
    transcripts_root, index_root = _default_fixture(tmp_path)
    # Tamper a checksummed file without re-signing metadata.
    (index_root / "vectors.npy").write_bytes(b"corrupt")
    with pytest.raises(Exception):  # noqa: B017 - checksum failure raises IndexArtifactError
        validate_asr_source(transcripts_root, index_root, ["L01_V001"])


def test_fingerprints_are_deterministic_across_repeated_validation(tmp_path: Path) -> None:
    transcripts_root, index_root = _default_fixture(tmp_path)
    first = validate_asr_source(transcripts_root, index_root, ["L01_V001", "L01_V002"])
    second = validate_asr_source(transcripts_root, index_root, ["L01_V001", "L01_V002"])
    assert first.transcript_fingerprint == second.transcript_fingerprint
    assert first.index_fingerprint == second.index_fingerprint
