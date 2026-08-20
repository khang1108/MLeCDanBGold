"""Bounded scheduling tests for the Thunder S3 ASR worker."""

from __future__ import annotations

from scripts.prepare_s3_asr_worker import _artifact_keys, _chunks
from hcmai.data.s3 import S3VideoObject


def _source(name: str, size: int) -> S3VideoObject:
    return S3VideoObject(
        key=f"data/{name}.mp4",
        size=size,
        etag="etag",
        last_modified_ns=1,
    )


def test_chunks_bound_download_bytes_and_retain_order() -> None:
    values = [_source("L21_V001", 6), _source("L21_V002", 4), _source("L21_V003", 7)]

    result = list(_chunks(values, max_bytes=10))

    assert [[item.video_id for item in chunk] for chunk in result] == [
        ["L21_V001", "L21_V002"],
        ["L21_V003"],
    ]


def test_single_large_video_is_not_dropped() -> None:
    source = _source("L21_V001", 20)

    assert list(_chunks([source], max_bytes=10)) == [[source]]


def test_artifact_keys_are_grouped_by_video_id() -> None:
    assert _artifact_keys("artifacts/production/transcripts", "L21_V001") == (
        "artifacts/production/transcripts/L21/L21_V001.parquet",
        "artifacts/production/transcripts/L21/L21_V001.manifest.json",
    )
