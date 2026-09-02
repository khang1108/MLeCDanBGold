"""Tests for reproducible Filter catalog latency/RSS measurement."""

from __future__ import annotations

import json
import sqlite3

from pathlib import Path

import pytest

from hcmai.filtering.schema import CATALOG_SCHEMA_VERSION, create_catalog_schema
from scripts.benchmark_filter_catalog import main


def _write_catalog(path: Path) -> None:
    """Write two canonical rows for fast deterministic benchmark tests."""

    connection = sqlite3.connect(path)
    create_catalog_schema(connection)
    connection.execute(
        """
        INSERT INTO catalog_metadata (
            id, schema_version, catalog_version, built_at, frame_count,
            source_lineage_json, title_available, caption_available,
            ocr_available, objects_available, asr_available
        ) VALUES (1, ?, 'benchmark-fixture-v1', '2026-09-02T00:00:00Z', 2, '{}', 1, 0, 0, 1, 0)
        """,
        (CATALOG_SCHEMA_VERSION,),
    )
    connection.executemany(
        """
        INSERT INTO frames (
            frame_id, video_id, frame_idx, timestamp_ms, folder_id,
            title, title_norm, caption, caption_norm, ocr, ocr_norm, asr, asr_norm
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL)
        """,
        [
            ("L21_V001_f1", "L21_V001", 25, 1000, "L21", "Red", "red"),
            ("L22_V001_f1", "L22_V001", 25, 1000, "L22", "Blue", "blue"),
        ],
    )
    connection.execute(
        "INSERT INTO frame_objects(frame_id, label_norm, object_count) VALUES (?, ?, ?)",
        ("L21_V001_f1", "person", 3),
    )
    connection.commit()
    connection.close()


@pytest.mark.parametrize("concurrency", [1, 4])
def test_benchmark_prints_stable_latency_and_rss_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    concurrency: int,
) -> None:
    """Record comparable metrics without changing Filter runtime behavior."""

    catalog_path = tmp_path / "filter.sqlite"
    _write_catalog(catalog_path)
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps(
            [
                {"name": "global-title", "request": {"metadata_filters": {"title": "red"}}},
                {"name": "folder-object", "request": {"folder_id": "L21", "metadata_filters": {"objects": {"person": 3}}}},
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--catalog", str(catalog_path),
            "--queries", str(queries_path),
            "--concurrency", str(concurrency),
            "--samples", "4",
            "--warmups", "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == "filter-benchmark-v1"
    assert payload["catalog_version"] == "benchmark-fixture-v1"
    assert payload["catalog_size_bytes"] > 0
    assert payload["concurrency"] == concurrency
    assert payload["samples_per_case"] == 4
    assert payload["warmups_per_case"] == 1
    assert payload["rss_delta_kib"] >= 0
    assert [case["name"] for case in payload["cases"]] == [
        "global-title", "folder-object"
    ]
    assert all(case["sample_count"] == 4 for case in payload["cases"])
    assert all(case["error_count"] == 0 for case in payload["cases"])
    assert all(case["p95_ms"] >= case["p50_ms"] >= 0 for case in payload["cases"])


@pytest.mark.parametrize(
    ("arguments", "query_payload"),
    [(["--concurrency", "0"], [{"name": "all", "request": {}}]),
     (["--samples", "0"], [{"name": "all", "request": {}}]),
     ([], []),
     ([], [{"name": "", "request": {}}])],
)
def test_benchmark_rejects_invalid_resource_or_query_configuration(
    tmp_path: Path,
    arguments: list[str],
    query_payload: list[dict[str, object]],
) -> None:
    """Fail before measurement when a run would be meaningless."""

    catalog_path = tmp_path / "filter.sqlite"
    _write_catalog(catalog_path)
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps(query_payload), encoding="utf-8")

    with pytest.raises(SystemExit):
        main(
            [
                "--catalog", str(catalog_path),
                "--queries", str(queries_path),
                *arguments,
            ]
        )
