"""Shared S3 staging lifecycle tests for composed preparation consumers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from hcmai.common.config import ASRConfig
from hcmai.common.schemas import FrameRecord, TranscriptSegment
from hcmai.data.enrichment.transcripts.adapters.asr import ASRAdapter
from hcmai.data.enrichment.transcripts.pipeline import TranscriptService
from hcmai.data.preprocessing import S3PreprocessingConfig
from hcmai.data.s3 import list_video_objects, staged_video


class _Paginator:
    def __init__(self, client: _FakeS3) -> None:
        self.client = client

    def paginate(self, *, Bucket: str, Prefix: str):
        assert Bucket == self.client.bucket
        yield {
            "Contents": [
                {
                    "Key": key,
                    "Size": len(value),
                    "ETag": '"source-etag"',
                    "LastModified": datetime(2026, 8, 13, tzinfo=UTC),
                }
                for key, value in self.client.objects.items()
                if key.startswith(Prefix)
            ]
        }


class _FakeS3:
    bucket = "hcmai-dataset"

    def __init__(self) -> None:
        self.objects = {"videos/L21_V001.mp4": b"newest-s3-video"}
        self.downloads: list[str] = []

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        assert bucket == self.bucket
        self.downloads.append(key)
        Path(filename).write_bytes(self.objects[key])


class _FakeASR:
    def __init__(self, frame_output: Path) -> None:
        self.frame_output = frame_output
        self.config = ASRConfig(device="cpu")
        self.resolved_revision = self.config.revision
        self.paths: list[Path] = []

    def transcribe(self, path: Path, video_id: str) -> list[TranscriptSegment]:
        assert path.is_file()
        assert self.frame_output.is_file()
        self.paths.append(path)
        return [TranscriptSegment(
            segment_id=f"{video_id}_segment_000000",
            video_id=video_id,
            segment_index=0,
            start_ms=0,
            end_ms=800,
            text="S3 transcript",
            language="vi",
        )]


def test_one_download_serves_frames_and_transcript_before_cleanup(
    tmp_path: Path,
) -> None:
    client = _FakeS3()
    storage = S3PreprocessingConfig(
        bucket=client.bucket,
        videos_prefix="videos",
        staging_root=tmp_path / "staging",
    )
    source = list_video_objects(client, storage)[0]
    frames_path = tmp_path / "artifacts/frame_store/frames.parquet"
    transcript_root = tmp_path / "artifacts/enrichment/transcripts"
    asr = _FakeASR(frames_path)
    service = TranscriptService(asr=cast(ASRAdapter, asr))

    with staged_video(client, storage, source) as staged:
        staged_path = staged
        frame = FrameRecord(
            frame_id="L21_V001_frame_000000000",
            video_id="L21_V001",
            frame_idx=0,
            timestamp_ms=0,
            image_path="images/L21_V001.jpg",
            width=8,
            height=8,
        )
        frames_path.parent.mkdir(parents=True)
        pd.DataFrame([frame.model_dump(mode="python")]).to_parquet(
            frames_path, index=False
        )

        transcript_path, count = service.prepare_video(
            staged, transcript_root
        )

        assert staged.is_file()
        assert frames_path.is_file()
        assert transcript_path.is_file()
        assert count == 1
        assert asr.paths == [staged]

    assert client.downloads == ["videos/L21_V001.mp4"]
    assert not staged_path.exists()
    assert frames_path.is_file()
    assert transcript_path.is_file()
