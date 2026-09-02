"""Tests for exact deterministic Filter SQL and pagination semantics."""

from __future__ import annotations

import sqlite3

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hcmai.api.contracts import FilterRequest
from hcmai.filtering.catalog import FilterCatalog
from hcmai.filtering.schema import CATALOG_SCHEMA_VERSION, create_catalog_schema
from hcmai.filtering.service import FilterService


def _write_service_catalog(
    path: Path,
    *,
    caption_available: bool = True,
) -> None:
    """Write six deliberately unordered rows with hand-checkable matches."""

    rows = [
        (
            "L22_V002_f2", "L22_V002", 50, 2000, "L22",
            "Other", "other", "quiet road", "quiet road", None, None,
            None, None,
        ),
        (
            "L21_V001_f3", "L21_V001", 75, 1000, "L21",
            "Áo đỏ", "ao do", "person near car", "person near car",
            "STOP", "stop", "Xin chào", "xin chao",
        ),
        (
            "L21_V001_f1", "L21_V001", 25, 1000, "L21",
            "Áo đỏ", "ao do", "person near car", "person near car",
            "STOP", "stop", "Xin chào", "xin chao",
        ),
        (
            "L21_V001_f4", "L21_V001", 100, 3000, "L21",
            "Áo đỏ", "ao do", None, None, "GO", "go", None, None,
        ),
        (
            "L21_V001_f2", "L21_V001", 50, 1000, "L21",
            "Áo đỏ", "ao do", "person near bicycle", "person near bicycle",
            "STOP", "stop", "Xin chào", "xin chao",
        ),
        (
            "L22_V002_f1", "L22_V002", 25, 500, "L22",
            "Other", "other", "city car", "city car", None, None,
            None, None,
        ),
    ]
    connection = sqlite3.connect(path)
    create_catalog_schema(connection)
    connection.execute(
        """
        INSERT INTO catalog_metadata (
            id, schema_version, catalog_version, built_at, frame_count,
            source_lineage_json, title_available, caption_available,
            ocr_available, objects_available, asr_available
        ) VALUES (1, ?, 'service-fixture-v1', '2026-09-02T00:00:00Z', 6, '{}', 1, ?, 1, 1, 1)
        """,
        (CATALOG_SCHEMA_VERSION, int(caption_available)),
    )
    connection.executemany(
        """
        INSERT INTO frames (
            frame_id, video_id, frame_idx, timestamp_ms, folder_id,
            title, title_norm, caption, caption_norm, ocr, ocr_norm, asr, asr_norm
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.executemany(
        "INSERT INTO frame_objects(frame_id, label_norm, object_count) VALUES (?, ?, ?)",
        [
            ("L21_V001_f1", "car", 1),
            ("L21_V001_f1", "person", 3),
            ("L21_V001_f2", "bicycle", 1),
            ("L21_V001_f2", "person", 3),
            ("L21_V001_f3", "car", 1),
            ("L21_V001_f3", "person", 1),
            ("L22_V002_f1", "car", 1),
        ],
    )
    connection.commit()
    connection.close()


@pytest.fixture
def service(tmp_path: Path) -> FilterService:
    """Open a production-shaped service over the tiny deterministic catalog."""

    path = tmp_path / "service.sqlite"
    _write_service_catalog(path)
    value = FilterService(FilterCatalog.open(path, pool_size=4))
    yield value
    value.close()


def test_filter_combines_normalized_text_and_exact_objects(service: FilterService) -> None:
    """Require every populated text and exact-count predicate to match."""

    response = service.filter(
        FilterRequest(
            metadata_filters={
                "title": "áo ĐỎ",
                "caption": "near car",
                "ocr": "stop",
                "asr": "xin chào",
                "objects": {"person": 3, "car": 1},
            }
        )
    )

    assert [row.frame_id for row in response.results] == ["L21_V001_f1"]
    assert response.results[0].objects == {"car": 1, "person": 3}


def test_filter_uses_exact_counts_not_minimum_counts(service: FilterService) -> None:
    """A request for one person must not match a frame containing three."""

    response = service.filter(
        FilterRequest(metadata_filters={"objects": {"person": 1}})
    )

    assert [row.frame_id for row in response.results] == ["L21_V001_f3"]


def test_filter_scopes_by_exact_folder_and_video(service: FilterService) -> None:
    """Apply organizer scope before stable pagination without rewriting IDs."""

    folder = service.filter(FilterRequest(folder_id="L22"))
    video = service.filter(FilterRequest(video_id="L21_V001"))
    incompatible = service.filter(
        FilterRequest(folder_id="L22", video_id="L21_V001")
    )

    assert [row.frame_id for row in folder.results] == [
        "L22_V002_f1", "L22_V002_f2"
    ]
    assert video.total_results == 4
    assert incompatible.total_results == 0


def test_available_but_missing_evidence_does_not_match(service: FilterService) -> None:
    """Treat a frame-level null as absent evidence rather than a text match."""

    response = service.filter(
        FilterRequest(metadata_filters={"caption": "person"})
    )

    assert "L21_V001_f4" not in {row.frame_id for row in response.results}
    assert response.total_results == 3


def test_globally_unavailable_predicate_is_ignored_and_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep V1 usable when a whole optional modality was not built."""

    path = tmp_path / "no-caption.sqlite"
    _write_service_catalog(path, caption_available=False)
    service = FilterService(FilterCatalog.open(path, pool_size=1))

    with caplog.at_level("WARNING"):
        response = service.filter(
            FilterRequest(metadata_filters={"caption": "no row has this"})
        )

    assert response.total_results == 6
    assert "caption" in caplog.text
    assert "ignored" in caplog.text
    service.close()


def test_filter_stably_orders_and_paginates_true_totals(service: FilterService) -> None:
    """Use canonical order for repeatable pages, including equal timestamps."""

    page = service.filter(FilterRequest(frames_per_pages=2, page_id=2))
    beyond = service.filter(FilterRequest(frames_per_pages=2, page_id=4))

    assert page.total_results == 6
    assert page.total_pages == 3
    assert [row.frame_id for row in page.results] == [
        "L21_V001_f3", "L21_V001_f4"
    ]
    assert beyond.total_results == 6
    assert beyond.total_pages == 3
    assert beyond.results == []


def test_zero_matches_have_zero_pages(service: FilterService) -> None:
    """Represent an empty result set without inventing a first page."""

    response = service.filter(
        FilterRequest(metadata_filters={"title": "missing"})
    )

    assert response.total_results == 0
    assert response.total_pages == 0
    assert response.results == []


def test_ten_callers_share_only_four_catalog_connections(
    service: FilterService,
) -> None:
    """Serve concurrent readers while retaining the configured memory ceiling."""

    request = FilterRequest(metadata_filters={"objects": {"person": 3}})
    with ThreadPoolExecutor(max_workers=10) as executor:
        responses = list(executor.map(service.filter, [request] * 10))

    assert [response.model_dump() for response in responses] == [
        responses[0].model_dump()
    ] * 10
    assert service.catalog.created_connection_count == 4


def test_health_exposes_only_safe_catalog_facts(service: FilterService) -> None:
    """Publish readiness/version/count without leaking the SQLite path."""

    assert service.health() == {
        "ready": True,
        "catalog_version": "service-fixture-v1",
        "frame_count": 6,
    }
