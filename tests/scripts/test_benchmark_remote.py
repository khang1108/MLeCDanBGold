"""Unit tests for benchmark_remote payload building and QA safety."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation.benchmark_remote import build_items, build_items_from_test_set
from scripts.evaluation.query_test_set import build_query_test_set, write_query_test_set


def _setup_test_queries(root: Path) -> Path:
    split_dir = root / "002"
    gt_dir = split_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    # 1 KIS query
    (split_dir / "query-p2-1-kis.txt").write_text("Cảnh con đập trên cao.", encoding="utf-8")
    (gt_dir / "query-p2-1-kis.csv").write_text("L24_V035,642\n", encoding="utf-8")

    # 1 QA query
    (split_dir / "query-p2-7-qa.txt").write_text(
        "Đầu bếp nhồi gia vị vào bốn con cá. Đây là loài cá gì?",
        encoding="utf-8",
    )
    (gt_dir / "query-p2-7-qa.csv").write_text("L01_V001,120,Cá Sòng\n", encoding="utf-8")

    # 1 TRAKE query
    (split_dir / "query-p2-8-trake.txt").write_text(
        "Giới thiệu chung.\nE1: Cảnh quả xoài.\nE2: Cảnh quả mít.",
        encoding="utf-8",
    )
    (gt_dir / "query-p2-8-trake.csv").write_text("L27_V011,100,200\n", encoding="utf-8")

    return root


def test_build_items_from_test_set_omits_forbidden_fields(tmp_path: Path) -> None:
    root = _setup_test_queries(tmp_path / "queries")
    test_set = build_query_test_set(root, "002", ground_truth_source="human_verified")
    fixture_path = tmp_path / "query_002.json"
    write_query_test_set(fixture_path, test_set)

    items = build_items_from_test_set(fixture_path)
    assert len(items) == 3

    forbidden_keys = {"raw_query", "answer", "ground_truth", "source_sha256"}

    for query_id, query_file, payload, kind in items:
        # Check forbidden keys are never inside the request payload
        for forbidden in forbidden_keys:
            assert forbidden not in payload, f"{forbidden} leaked into payload for {query_id}"

        if kind == "qa":
            assert query_id == "query-p2-7-qa"
            assert payload["query"] == "Đầu bếp nhồi gia vị vào bốn con cá"
            assert "Đây là loài cá gì?" not in str(payload["query"])
            assert "Cá Sòng" not in str(payload)
        elif kind == "kis":
            assert query_id == "query-p2-1-kis"
            assert payload["query"] == "Cảnh con đập trên cao."
        elif kind == "trake":
            assert query_id == "query-p2-8-trake"
            assert payload["events"] == ["Cảnh quả xoài.", "Cảnh quả mít."]
            assert "query" not in payload


def test_build_items_from_test_set_filters_by_allowed_kinds(tmp_path: Path) -> None:
    root = _setup_test_queries(tmp_path / "queries")
    test_set = build_query_test_set(root, "002", ground_truth_source="human_verified")
    fixture_path = tmp_path / "query_002.json"
    write_query_test_set(fixture_path, test_set)

    items = build_items_from_test_set(fixture_path, allowed_kinds={"kis"})
    assert len(items) == 1
    assert items[0][3] == "kis"

    items_trake = build_items_from_test_set(fixture_path, allowed_kinds={"trake"})
    assert len(items_trake) == 1
    assert items_trake[0][3] == "trake"


def test_legacy_build_items_strips_qa_question(tmp_path: Path) -> None:
    root = _setup_test_queries(tmp_path / "queries")
    items = build_items(root, "002")

    qa_items = [item for item in items if item[3] == "qa"]
    assert len(qa_items) == 1
    query_id, query_file, payload, kind = qa_items[0]

    assert payload["query"] == "Đầu bếp nhồi gia vị vào bốn con cá"
    assert "Đây là loài cá gì?" not in str(payload["query"])
    assert "raw_query" not in payload
    assert "answer" not in payload
