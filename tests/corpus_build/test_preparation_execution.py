from __future__ import annotations

import threading
from pathlib import Path

from hcmai.data.corpus_build.execution import prepare_cached_sources
from hcmai.data.s3 import S3VideoObject


def _source(index: int) -> S3VideoObject:
    return S3VideoObject(
        key=f"data/V{index:03d}.mp4",
        size=1,
        etag=f"etag-{index}",
        last_modified_ns=index,
    )


def test_first_video_warms_serially_then_lanes_overlap(tmp_path: Path) -> None:
    sources = [_source(1), _source(2), _source(3)]
    paths = {
        source.video_id: tmp_path / f"{source.video_id}.mp4"
        for source in sources
    }
    for path in paths.values():
        path.write_bytes(b"x")
    frame_entered = threading.Event()
    asr_entered = threading.Event()
    events: list[str] = []

    def frame(path: Path, source: S3VideoObject) -> str:
        events.append(f"frame:{source.video_id}")
        if source.video_id == "V002":
            frame_entered.set()
            assert asr_entered.wait(timeout=2)
        return source.video_id

    def transcript(path: Path) -> Path:
        events.append(f"asr:{path.stem}")
        if path.stem == "V002":
            asr_entered.set()
            assert frame_entered.wait(timeout=2)
        return path

    tables = prepare_cached_sources(
        sources,
        resolve=lambda source: paths[source.video_id],
        prepare_frame=frame,
        prepare_transcript=transcript,
        frame_pending=True,
        asr_pending=True,
        overlap=True,
    )

    assert events[:2] == ["frame:V001", "asr:V001"]
    assert tables == ["V001", "V002", "V003"]


def test_disabled_overlap_preserves_per_video_order(tmp_path: Path) -> None:
    sources = [_source(1), _source(2)]
    events: list[str] = []

    prepare_cached_sources(
        sources,
        resolve=lambda source: tmp_path / f"{source.video_id}.mp4",
        prepare_frame=lambda path, source: events.append(
            f"frame:{source.video_id}"
        ),
        prepare_transcript=lambda path: events.append(f"asr:{path.stem}"),
        frame_pending=True,
        asr_pending=True,
        overlap=False,
    )

    assert events == ["frame:V001", "asr:V001", "frame:V002", "asr:V002"]
