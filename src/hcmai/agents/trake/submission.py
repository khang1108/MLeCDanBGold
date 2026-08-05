"""Ranking, video diversification, and TRAKE submission CSV export."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from hcmai.common.utils.logging import get_logger

from .align import TrakePath, align_video
from .shortlist import VideoEventScores

logger = get_logger(__name__)


def rank_paths(
    videos: Sequence[VideoEventScores],
    lambda_gap: float = 1e-5,
    max_rows: int = 100,
) -> list[TrakePath]:
    """Rank aligned paths, one video per row before any video repeats.

    Every video contributes its best path first, sorted by ``score``, because
    a wrong video scores zero for the whole row and ``R@k`` only keeps the best
    row inside each cutoff. Only once the best paths run out do leading videos
    contribute their second-best path, and so on, until ``max_rows`` rows exist.

    Args:
        videos: Shortlisted videos with their event/frame score matrices.
        lambda_gap: Time-gap penalty per millisecond, passed to the aligner.
        max_rows: Official per-query row limit.

    Returns:
        At most ``max_rows`` paths, best first. Fewer only when the shortlisted
        videos cannot yield that many increasing paths.
    """
    if not videos:
        return []
    depth = -(-max_rows // len(videos))
    per_video = [align_video(video, lambda_gap, depth) for video in videos]
    rows: list[TrakePath] = []
    for level in range(depth):
        rows.extend(
            sorted(
                (paths[level] for paths in per_video if len(paths) > level),
                key=lambda path: path.score,
                reverse=True,
            )
        )
        if len(rows) >= max_rows:
            break
    return rows[:max_rows]


def write_submission(rows: Sequence[TrakePath], output_path: str | Path) -> Path:
    """Write ranked TRAKE paths as one official per-query CSV.

    Emits headerless UTF-8 rows of ``<video_name>,<frame_1>,...,<frame_N>``.
    ``frame_idx`` values already come from the canonical frame mapping, so this
    only drops the ``.mp4`` extension to get the official video name.

    Args:
        rows: Ranked paths from :func:`rank_paths`.
        output_path: Destination file, for example
            ``submission/query-1-trake.csv``. Parent directories are created.

    Returns:
        The written path.

    Raises:
        ValueError: If the rows disagree on event count, since a row with the
            wrong column count is scored as invalid.
    """
    counts = {len(row.frame_idx) for row in rows}
    if len(counts) > 1:
        raise ValueError(f"rows mix event counts: {sorted(counts)}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(
            [row.video_id.removesuffix(".mp4"), *row.frame_idx] for row in rows
        )
    if len(rows) < 100:
        logger.warning(
            "TRAKE submission %s has %d rows, under the 100-row budget",
            path,
            len(rows),
        )
    return path
