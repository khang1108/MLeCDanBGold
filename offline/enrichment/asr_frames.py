"""Project segment ASR transcripts onto canonical frames for the BM25 asr field.

Transcripts are segment-native while the BM25 builder joins on ``frame_id``.
Timeline selection is delegated to the shared ``SegmentFrameProjector``; this
module never invents frame identity and never rewrites transcript text.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hcmai.retrieval.retriever.segment.projector import SegmentFrameProjector

FRAMES = Path("artifacts/custom-raw1fps-v1/frame_store/frames.parquet")
TRANSCRIPTS = Path("artifacts/enrichment/transcripts")
OUTPUT = Path("artifacts/enrichment/asr/frame_enrichment.parquet")
SEGMENT_COLUMNS = ("video_id", "segment_index", "start_ms", "end_ms", "text", "status")


def load_segments(root: Path) -> pd.DataFrame:
    """Read every completed transcript segment that carries text."""

    paths = sorted(root.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no transcript parquet under {root}")
    table = pd.concat(
        [pd.read_parquet(path, columns=list(SEGMENT_COLUMNS)) for path in paths],
        ignore_index=True,
    )
    table = table[table["status"] == "completed"].copy()
    table["text"] = table["text"].astype(str).str.strip()
    table = table[table["text"].astype(bool)]
    return table.sort_values(["video_id", "segment_index"]).reset_index(drop=True)


@dataclass(frozen=True)
class _Frame:
    """The four frame fields the projector reads, without a corpus import."""

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int


def _frames_of(frames: pd.DataFrame) -> list[_Frame]:
    """Adapt canonical frame rows to the projector's runtime model."""

    return [
        _Frame(
            frame_id=str(row.frame_id),
            video_id=str(row.video_id),
            frame_idx=int(row.frame_idx),
            timestamp_ms=int(row.timestamp_ms),
        )
        for row in frames.itertuples()
    ]


def project_segments(
    frames: pd.DataFrame,
    segments: pd.DataFrame,
    max_gap_ms: int = 5_000,
) -> pd.DataFrame:
    """Return one ``frame_id, asr_text`` row per frame any segment speaks over."""

    by_video: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for video_id, group in frames.groupby("video_id", sort=False):
        ordered = group.sort_values("timestamp_ms")
        by_video[str(video_id)] = (
            ordered["timestamp_ms"].to_numpy(),
            ordered["frame_id"].to_numpy(),
        )

    hits: list[tuple[str, int, str]] = []
    missed: list[tuple[int, str, int, int, str]] = []
    for order, row in enumerate(segments.itertuples()):
        entry = by_video.get(str(row.video_id))
        if entry is None:
            continue
        timestamps, frame_ids = entry
        low = int(np.searchsorted(timestamps, row.start_ms, "left"))
        high = int(np.searchsorted(timestamps, row.end_ms, "right"))
        if high > low:
            hits.extend((str(frame_id), order, row.text) for frame_id in frame_ids[low:high])
        else:
            missed.append(
                (order, str(row.video_id), int(row.start_ms), int(row.end_ms), row.text)
            )

    if missed:
        videos = {video_id for _, video_id, _, _, _ in missed}
        projector = SegmentFrameProjector(
            _frames_of(frames[frames["video_id"].isin(videos)]), max_gap_ms
        )
        for order, video_id, start_ms, end_ms, text in missed:
            projection = projector.project(video_id, start_ms=start_ms, end_ms=end_ms)
            if projection is not None:
                hits.append((str(projection.frame_id), order, text))

    table = pd.DataFrame(hits, columns=["frame_id", "order", "text"])
    table = table.sort_values(["frame_id", "order"])
    grouped = table.groupby("frame_id", sort=False)["text"].apply(" ".join)
    return grouped.rename("asr_text").reset_index()


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=FRAMES)
    parser.add_argument("--transcripts", type=Path, default=TRANSCRIPTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-gap-ms", type=int, default=5_000)
    return parser


def main() -> None:
    """Write the frame-native ASR artifact consumed by the BM25 builder."""

    arguments = _parser().parse_args()
    frames = pd.read_parquet(
        arguments.frames, columns=["frame_id", "video_id", "frame_idx", "timestamp_ms"]
    )
    segments = load_segments(arguments.transcripts)
    projected = project_segments(frames, segments, arguments.max_gap_ms)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    projected.to_parquet(arguments.output, index=False)
    print(f"{len(segments)} segments -> {len(projected)}/{len(frames)} frames")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
