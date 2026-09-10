"""Tests for bounded-memory technical-report artifact statistics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs/reports/technical/tools/summarize_artifacts.py"
)
_SPEC = importlib.util.spec_from_file_location("summarize_report_artifacts", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_stream_frame_distribution_aggregates_multiple_batches(tmp_path: Path) -> None:
    """Frame statistics retain only counters while crossing batch boundaries."""

    path = tmp_path / "frames.parquet"
    pq.write_table(
        pa.table(
            {
                "video_id": [
                    "L21_V001",
                    "L21_V001",
                    "L21_V002",
                    "L22_V001",
                    "L22_V001",
                ]
            }
        ),
        path,
    )

    frames, batches, collections = _MODULE.stream_frame_distribution(path, batch_size=2)

    assert frames == 5
    assert batches == 3
    assert collections == {
        "L21": {"videos": 2, "frames": 3},
        "L22": {"videos": 1, "frames": 2},
    }


def test_stream_row_count_uses_bounded_batches(tmp_path: Path) -> None:
    """A detail artifact is counted batch by batch and checked against its footer."""

    path = tmp_path / "regions.parquet"
    pq.write_table(pa.table({"region_id": list(range(7)), "text": ["x"] * 7}), path)

    rows, batches, footer_rows = _MODULE.stream_row_count(
        path,
        "region_id",
        batch_size=3,
    )

    assert (rows, batches, footer_rows) == (7, 3, 7)


def test_render_tex_precomputes_safe_bar_widths() -> None:
    """Large frame totals are normalized before TikZ parses bar dimensions."""

    statistics = {
        "summary": {
            "videos": 586,
            "frames": 288_342,
            "sampled_timeline_hours": 80.1,
            "ocr_regions": 7,
            "object_detections": 11,
            "asr_segments": 3,
        },
        "artifacts": {
            "captions": {"rows": 288_342},
            "ocr_frames": {"rows": 288_342},
            "object_frames": {"rows": 288_342},
            "frame_context": {"rows": 288_342},
        },
        "collections": [
            {"collection": "L25", "videos": 88, "frames": 130_481},
            {"collection": "L26", "videos": 498, "frames": 157_861},
        ],
    }

    rendered = _MODULE.render_tex(statistics)

    assert r"\ArtifactCollectionBar{45}{L25}{88}{61.9917}{130,481}" in rendered
    assert r"\ArtifactCollectionBar{40}{L26}{498}{75.0000}{157,861}" in rendered


def test_rendered_outputs_are_deterministic(tmp_path: Path) -> None:
    """Repeated writes of the same summary produce byte-identical report inputs."""

    statistics = {
        "summary": {
            "videos": 2,
            "frames": 5,
            "sampled_timeline_hours": 0.0,
            "ocr_regions": 7,
            "object_detections": 11,
            "asr_segments": 3,
        },
        "artifacts": {
            "captions": {"rows": 5},
            "ocr_frames": {"rows": 5},
            "object_frames": {"rows": 5},
            "frame_context": {"rows": 5},
        },
        "collections": [
            {"collection": "L21", "videos": 2, "frames": 5},
        ],
    }

    paths = _MODULE.write_outputs(statistics, tmp_path)
    first = tuple(path.read_bytes() for path in paths)
    _MODULE.write_outputs(statistics, tmp_path)
    second = tuple(path.read_bytes() for path in paths)

    assert first == second
    assert all(path.stat().st_mode & 0o777 == 0o644 for path in paths)
    assert b"\\ArtifactCollectionBar{45}{L21}{2}{75.0000}{5}" in second[1]
