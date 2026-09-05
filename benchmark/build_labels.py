"""Convert organizer submission CSVs into benchmark labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

FRAMES = Path("artifacts/custom-raw1fps-v1/frame_store/frames.parquet")
GROUND_TRUTH = Path("benchmark/grouth_truth")


class FrameLookup:
    """Resolve a BTC frame_idx to the nearest sampled corpus frame."""

    def __init__(self, frames: pd.DataFrame) -> None:
        self.by_video = {
            video_id: group.sort_values("frame_idx").reset_index(drop=True)
            for video_id, group in frames.groupby("video_id", sort=False)
        }

    def nearest(self, video_id: str, frame_idx: int) -> dict[str, Any]:
        """Return the corpus frame whose BTC coordinate is closest to frame_idx."""

        group = self.by_video.get(video_id)
        if group is None:
            raise KeyError(f"video absent from corpus: {video_id}")
        values = group["frame_idx"].to_numpy()
        position = int(np.searchsorted(values, frame_idx))
        position = min(position, len(values) - 1)
        if position and abs(values[position - 1] - frame_idx) < abs(values[position] - frame_idx):
            position -= 1
        row = group.iloc[position]
        return {
            "video_id": video_id,
            "frame_id": str(row["frame_id"]),
            "frame_idx": int(row["frame_idx"]),
            "timestamp_ms": int(row["timestamp_ms"]),
            "gap_frames": abs(int(row["frame_idx"]) - frame_idx),
        }


def _round_tag(path: Path) -> str:
    """Name the exam round owning one submission file."""

    return path.parents[2].name.split("-", 1)[0]


def _read_rows(path: Path) -> list[list[str]]:
    """Read non-empty CSV rows from one submission file."""

    with path.open(encoding="utf-8-sig") as handle:
        return [row for row in csv.reader(handle) if row and row[0].strip()]


def _label(
    path: Path, lookup: FrameLookup, source: str, top_k: int | None
) -> dict[str, Any] | None:
    """Build one label record from a submission file."""

    # Rounds reuse the same query-p2-N-kis names for different questions, so the
    # round has to be part of the identity or later rounds silently replace earlier ones.
    query_id = f"{_round_tag(path)}:{path.stem}"
    task = path.stem.rsplit("-", 1)[1]
    rows = _read_rows(path)[:top_k]
    if not rows:
        return None

    label: dict[str, Any] = {
        "query_id": query_id,
        "task": task,
        "source": source,
        "video_id": rows[0][0].strip(),
    }

    if task == "trake":
        label["paths"] = [
            {
                "video_id": row[0].strip(),
                "events": [lookup.nearest(row[0].strip(), int(value)) for value in row[1:]],
            }
            for row in rows
        ]
        return label

    candidates = [lookup.nearest(row[0].strip(), int(row[1])) for row in rows]
    label["candidates"] = candidates
    videos = {candidate["video_id"] for candidate in candidates}
    if len(videos) == 1:
        timestamps = sorted(int(candidate["timestamp_ms"]) for candidate in candidates)
        label["start_ms"] = timestamps[0]
        label["end_ms"] = timestamps[-1]
    return label


def build(
    frames_path: Path,
    sources: list[tuple[Path, str, int | None]],
    output: Path,
) -> list[dict[str, Any]]:
    """Write labels.jsonl for every submission file found under the sources."""

    frames = pd.read_parquet(
        frames_path, columns=["frame_id", "video_id", "frame_idx", "timestamp_ms"]
    )
    lookup = FrameLookup(frames)
    labels: list[dict[str, Any]] = []
    for directory, source, top_k in sources:
        for path in sorted(directory.glob("*.csv")):
            label = _label(path, lookup, source, top_k)
            if label is not None:
                labels.append(label)
    labels.sort(key=lambda record: str(record["query_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label, ensure_ascii=False) + "\n")
    return labels


def write_queries(labels: list[dict[str, Any]], directories: list[Path], output: Path):
    """Write the query file consumed by scripts/evaluate_temporal_p0.py."""

    texts: dict[str, str] = {}
    for directory in directories:
        for path in sorted(directory.parents[1].glob("query-*.txt")):
            texts[f"{path.parent.name.split('-', 1)[0]}:{path.stem}"] = path.read_text(
                encoding="utf-8"
            ).strip()
    items = [
        {
            "query_id": label["query_id"],
            "query": texts[str(label["query_id"])],
            "target_video_id": label["video_id"],
        }
        for label in labels
        if str(label["query_id"]) in texts
    ]
    output.write_text(
        yaml.safe_dump({"queries": items}, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return items


def main() -> None:
    """Build labels from the organizer submission directories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=FRAMES)
    parser.add_argument(
        "--curated",
        type=Path,
        default=GROUND_TRUTH / "SOTUYEN1-bo-de-thi/submission_final/submission",
        help="submissions whose every row is a verified answer",
    )
    parser.add_argument(
        "--ranked",
        type=Path,
        nargs="+",
        default=[
            GROUND_TRUTH / "SOTUYEN2-bo-de-thi/submission_final/submission",
            GROUND_TRUTH / "SOTUYEN3-bo-de-thi/submission_final/submission",
        ],
        help="shotgun submissions; only the first --top-k rows are kept",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("benchmark/labels.jsonl"))
    parser.add_argument("--queries-out", type=Path, default=Path("benchmark/queries.yaml"))
    arguments = parser.parse_args()
    sources: list[tuple[Path, str, int | None]] = [(arguments.curated, "curated", None)]
    sources += [(path, "top10", arguments.top_k) for path in arguments.ranked]
    labels = build(arguments.frames, sources, arguments.output)
    queries = write_queries(labels, [path for path, _, _ in sources], arguments.queries_out)

    counts: dict[tuple[str, str], int] = {}
    for label in labels:
        key = (str(label["source"]), str(label["task"]))
        counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(f"{key[0]:8s} {key[1]:6s} {counts[key]}")
    print(f"{len(labels)} labels -> {arguments.output}")
    print(f"{len(queries)} queries -> {arguments.queries_out}")


if __name__ == "__main__":
    main()
