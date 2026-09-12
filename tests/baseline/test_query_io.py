"""Tests for baseline query parsing and immutable data contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from baseline.contracts import BaselinePath, MethodOptions, QueryCase
from baseline.query_io import load_query_cases


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_query_cases_parses_kis_and_trake_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "queries"
    _write(root / "002" / "b-trake.txt", "These scenes are consecutive.\nE1: first event\nCảnh 2: second event\n")
    _write(root / "002" / "a-kis.txt", "first moment\nsecond moment\n")
    _write(root / "002" / "c-qa.txt", "question scene")

    cases = load_query_cases(root, ["002"], {"kis", "trake"}, None)

    assert [Path(case.query_file).name for case in cases] == ["a-kis.txt", "b-trake.txt"]
    assert cases[0] == QueryCase(
        query_file=str(root / "002" / "a-kis.txt"),
        kind="kis",
        raw_query="first moment\nsecond moment",
        events=("first moment", "second moment"),
    )
    assert cases[1].events == ("first event", "second event")


def test_load_query_cases_keeps_raw_qa_but_omits_question_from_retrieval(tmp_path: Path) -> None:
    root = tmp_path / "queries"
    raw_query = "Cảnh đầu bếp nhồi gia vị vào bốn con cá. Đây là loài cá gì?"
    _write(root / "002" / "a-qa.txt", raw_query)

    cases = load_query_cases(root, ["002"], {"qa"}, None)

    assert cases[0].raw_query == raw_query
    assert cases[0].events == ("Cảnh đầu bếp nhồi gia vị vào bốn con cá",)
    assert cases[0].retrieval_query == "Cảnh đầu bếp nhồi gia vị vào bốn con cá"


def test_load_query_cases_filters_before_applying_max_queries(tmp_path: Path) -> None:
    root = tmp_path / "queries"
    _write(root / "001" / "a-kis.txt", "ignored split")
    _write(root / "002" / "a-qa.txt", "ignored kind")
    _write(root / "002" / "b-kis.txt", "selected first")
    _write(root / "002" / "c-kis.txt", "selected second")

    cases = load_query_cases(root, ["002"], {"kis"}, 1)

    assert [Path(case.query_file).name for case in cases] == ["b-kis.txt"]


def test_load_query_cases_rejects_trake_file_without_events(tmp_path: Path) -> None:
    root = tmp_path / "queries"
    path = root / "002" / "broken-trake.txt"
    _write(path, "These scenes are consecutive, but no event lines follow.")

    with pytest.raises(ValueError, match=r"broken-trake\.txt.*no TRAKE events"):
        load_query_cases(root, ["002"], {"trake"}, None)


def test_load_query_cases_rejects_invalid_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one split"):
        load_query_cases(tmp_path, [], {"kis"}, None)
    with pytest.raises(ValueError, match="unsupported query kinds"):
        load_query_cases(tmp_path, ["002"], {"other"}, None)
    with pytest.raises(ValueError, match="max_queries"):
        load_query_cases(tmp_path, ["002"], {"kis"}, 0)


def test_baseline_path_validates_identity_arrays_and_order() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        BaselinePath("v1", 1.0, ("f1",), (), (0,), None)
    with pytest.raises(ValueError, match="permutation"):
        BaselinePath("v1", 1.0, ("f1", "f2"), (1, 2), (0, 1), (0, 0))
    with pytest.raises(ValueError, match="finite"):
        BaselinePath("v1", float("nan"), ("f1",), (1,), (0,), None)


def test_method_options_validate_lattice_bounds() -> None:
    with pytest.raises(ValueError, match="candidates_per_event"):
        MethodOptions(candidates_per_event=0)
    with pytest.raises(ValueError, match="max_order_candidates"):
        MethodOptions(max_order_candidates=0)
