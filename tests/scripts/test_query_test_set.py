"""Tests for compiling competition query artifacts into a frozen test set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation import query_test_set as query_test_set_module
from scripts.evaluation.query_test_set import build_query_test_set


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_query_test_set_compiles_kis_query_and_labels(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        "L24_V035,642\nL24_V035,696\n",
    )

    test_set = build_query_test_set(root, "002", ground_truth_source="human_verified")

    assert test_set["schema_version"] == "hcmai-query-test-set-v1"
    assert test_set["split"] == "002"
    assert test_set["ground_truth_source"] == "human_verified"
    assert test_set["qa_mode"] == "retrieval_only"
    assert test_set["cases"] == [
        {
            "query_id": "query-p2-1-kis",
            "kind": "kis",
            "source_query_file": "002/query-p2-1-kis.txt",
            "raw_query": "Một cảnh cần tìm.",
            "retrieval_query": "Một cảnh cần tìm.",
            "events": ["Một cảnh cần tìm"],
            "answer": None,
            "ground_truth": [
                {"video_id": "L24_V035", "frame_idxs": [642]},
                {"video_id": "L24_V035", "frame_idxs": [696]},
            ],
        }
    ]


def test_build_query_test_set_omits_qa_question_but_keeps_answer_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "query"
    raw_query = "Đầu bếp nhồi gia vị vào bốn con cá. Đây là loài cá gì?"
    _write(root / "002" / "query-p2-7-qa.txt", raw_query)
    _write(
        root / "002" / "ground_truth" / "query-p2-7-qa.csv",
        "L01_V001,120,Cá Sòng\nL01_V002,240,Cá Sòng\n",
    )

    test_set = build_query_test_set(root, "002", ground_truth_source="human_verified")

    assert test_set["cases"] == [
        {
            "query_id": "query-p2-7-qa",
            "kind": "qa",
            "source_query_file": "002/query-p2-7-qa.txt",
            "raw_query": raw_query,
            "retrieval_query": "Đầu bếp nhồi gia vị vào bốn con cá",
            "events": ["Đầu bếp nhồi gia vị vào bốn con cá"],
            "answer": "Cá Sòng",
            "ground_truth": [
                {"video_id": "L01_V001", "frame_idxs": [120]},
                {"video_id": "L01_V002", "frame_idxs": [240]},
            ],
        }
    ]


def test_build_query_test_set_uses_explicit_trake_events_and_path_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "query"
    raw_query = (
        "Giới thiệu không dùng để retrieval.\n"
        "E1: Cảnh có trái sầu riêng.\n"
        "E2: Cảnh có trái măng cụt.\n"
        "E3: Cảnh có trái bưởi.\n"
        "E4: Cảnh có trái dâu bòn bon."
    )
    _write(root / "002" / "query-p2-8-trake.txt", raw_query)
    _write(
        root / "002" / "ground_truth" / "query-p2-8-trake.csv",
        "L27_V011,3858,3870,4012,4102\nL28_V015,12078,12115,12563,12766\n",
    )

    test_set = build_query_test_set(root, "002", ground_truth_source="human_verified")

    events = [
        "Cảnh có trái sầu riêng.",
        "Cảnh có trái măng cụt.",
        "Cảnh có trái bưởi.",
        "Cảnh có trái dâu bòn bon.",
    ]
    assert test_set["cases"] == [
        {
            "query_id": "query-p2-8-trake",
            "kind": "trake",
            "source_query_file": "002/query-p2-8-trake.txt",
            "raw_query": raw_query,
            "retrieval_query": " ".join(events),
            "events": events,
            "answer": None,
            "ground_truth": [
                {"video_id": "L27_V011", "frame_idxs": [3858, 3870, 4012, 4102]},
                {
                    "video_id": "L28_V015",
                    "frame_idxs": [12078, 12115, 12563, 12766],
                },
            ],
        }
    ]


def test_build_query_test_set_hashes_query_and_label_sources(tmp_path: Path) -> None:
    root = tmp_path / "query"
    query_path = root / "002" / "query-p2-1-kis.txt"
    _write(query_path, "Một cảnh cần tìm.")
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        "L24_V035,642\n",
    )

    first = build_query_test_set(root, "002", ground_truth_source="human_verified")
    second = build_query_test_set(root, "002", ground_truth_source="human_verified")
    _write(query_path, "Một cảnh khác cần tìm.")
    changed = build_query_test_set(root, "002", ground_truth_source="human_verified")

    assert len(first["source_sha256"]) == 64
    assert first["source_sha256"] == second["source_sha256"]
    assert changed["source_sha256"] != first["source_sha256"]


def test_build_query_test_set_orders_numbered_queries_naturally(tmp_path: Path) -> None:
    root = tmp_path / "query"
    for number in (10, 2):
        stem = f"query-p2-{number}-kis"
        _write(root / "002" / f"{stem}.txt", f"Query {number}.")
        _write(root / "002" / "ground_truth" / f"{stem}.csv", "L01_V001,1\n")

    test_set = build_query_test_set(root, "002", ground_truth_source="human_verified")

    assert [case["query_id"] for case in test_set["cases"]] == [
        "query-p2-2-kis",
        "query-p2-10-kis",
    ]


def test_build_query_test_set_rejects_query_without_ground_truth(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")

    with pytest.raises(ValueError, match="missing ground truth.*query-p2-1-kis.csv"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_build_query_test_set_rejects_ground_truth_without_query(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        "L24_V035,642\n",
    )

    with pytest.raises(ValueError, match="missing query.*query-p2-1-kis.txt"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_build_query_test_set_rejects_wrong_ground_truth_width(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        "L24_V035,642,unexpected\n",
    )

    with pytest.raises(ValueError, match="expected 2 columns.*row 1"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_build_query_test_set_rejects_duplicate_ground_truth_rows(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        "L24_V035,642\nL24_V035,642\n",
    )

    with pytest.raises(ValueError, match="duplicate ground truth.*row 2"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_build_query_test_set_rejects_inconsistent_qa_answers(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(
        root / "002" / "query-p2-7-qa.txt",
        "Đầu bếp nhồi gia vị vào cá. Đây là loài cá gì?",
    )
    _write(
        root / "002" / "ground_truth" / "query-p2-7-qa.csv",
        "L01_V001,120,Cá Sòng\nL01_V002,240,Cá hồi\n",
    )

    with pytest.raises(ValueError, match="QA answers must be identical"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


@pytest.mark.parametrize("frame_idx", ["not-an-int", "-1"])
def test_build_query_test_set_rejects_invalid_frame_indexes(
    tmp_path: Path,
    frame_idx: str,
) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        f"L24_V035,{frame_idx}\n",
    )

    with pytest.raises(ValueError, match="frame index must be a non-negative integer.*row 1"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_build_query_test_set_rejects_empty_ground_truth(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(root / "002" / "ground_truth" / "query-p2-1-kis.csv", "")

    with pytest.raises(ValueError, match="ground truth must contain at least one row"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_build_query_test_set_rejects_trake_without_explicit_events(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-8-trake.txt", "Chỉ có phần giới thiệu.")
    _write(
        root / "002" / "ground_truth" / "query-p2-8-trake.csv",
        "L27_V011,1,2,3,4\n",
    )

    with pytest.raises(ValueError, match="no TRAKE events found"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_build_query_test_set_rejects_qa_without_scene_text(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-7-qa.txt", "Đây là loài cá gì?")
    _write(
        root / "002" / "ground_truth" / "query-p2-7-qa.csv",
        "L01_V001,120,Cá Sòng\n",
    )

    with pytest.raises(ValueError, match="QA query must contain scene text before the question"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_build_query_test_set_rejects_non_chronological_trake_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "query"
    _write(
        root / "002" / "query-p2-8-trake.txt",
        "E1: Cảnh đầu.\nE2: Cảnh sau.",
    )
    _write(
        root / "002" / "ground_truth" / "query-p2-8-trake.csv",
        "L27_V011,200,100\n",
    )

    with pytest.raises(ValueError, match="TRAKE frame indexes must be chronological.*row 1"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_build_query_test_set_rejects_blank_video_id(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(root / "002" / "ground_truth" / "query-p2-1-kis.csv", ",642\n")

    with pytest.raises(ValueError, match="video_id must be non-empty.*row 1"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_build_query_test_set_rejects_blank_qa_answer(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(
        root / "002" / "query-p2-7-qa.txt",
        "Đầu bếp nhồi gia vị vào cá. Đây là loài cá gì?",
    )
    _write(
        root / "002" / "ground_truth" / "query-p2-7-qa.csv",
        "L01_V001,120,\n",
    )

    with pytest.raises(ValueError, match="QA answer must be non-empty"):
        build_query_test_set(root, "002", ground_truth_source="human_verified")


def test_write_and_load_query_test_set_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        "L24_V035,642\n",
    )
    value = build_query_test_set(root, "002", ground_truth_source="human_verified")
    output = tmp_path / "fixtures" / "query_002.json"

    query_test_set_module.write_query_test_set(output, value)

    assert query_test_set_module.load_query_test_set(output) == value
    assert output.read_bytes().endswith(b"\n")
    assert "Một cảnh cần tìm" in output.read_text(encoding="utf-8")


def test_load_query_test_set_rejects_unknown_schema_version(tmp_path: Path) -> None:
    fixture = tmp_path / "query_002.json"
    fixture.write_text(
        json.dumps({"schema_version": "unknown", "cases": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported query test-set schema"):
        query_test_set_module.load_query_test_set(fixture)


def test_load_query_test_set_rejects_qa_question_leakage(tmp_path: Path) -> None:
    root = tmp_path / "query"
    raw_query = "Đầu bếp nhồi gia vị vào cá. Đây là loài cá gì?"
    _write(root / "002" / "query-p2-7-qa.txt", raw_query)
    _write(
        root / "002" / "ground_truth" / "query-p2-7-qa.csv",
        "L01_V001,120,Cá Sòng\n",
    )
    value = build_query_test_set(root, "002", ground_truth_source="human_verified")
    value["cases"][0]["retrieval_query"] = raw_query
    fixture = tmp_path / "query_002.json"
    fixture.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="QA retrieval_query must equal joined events"):
        query_test_set_module.load_query_test_set(fixture)


def test_load_query_test_set_rejects_boolean_frame_indexes(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        "L24_V035,642\n",
    )
    value = build_query_test_set(root, "002", ground_truth_source="human_verified")
    value["cases"][0]["ground_truth"][0]["frame_idxs"] = [True]  # type: ignore[list-item]
    fixture = tmp_path / "query_002.json"
    fixture.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="frame_idx must be non-negative integer"):
        query_test_set_module.load_query_test_set(fixture)


def test_load_query_test_set_rejects_trake_event_count_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "query"
    raw_query = "E1: Sự kiện 1.\nE2: Sự kiện 2."
    _write(root / "002" / "query-p2-8-trake.txt", raw_query)
    _write(
        root / "002" / "ground_truth" / "query-p2-8-trake.csv",
        "L27_V011,100,200\n",
    )
    value = build_query_test_set(root, "002", ground_truth_source="human_verified")
    value["cases"][0]["ground_truth"][0]["frame_idxs"] = [100]  # Only 1 frame for 2 events
    fixture = tmp_path / "query_002.json"
    fixture.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="TRAKE ground_truth entry frame count \\(1\\) must match event count \\(2\\)"):
        query_test_set_module.load_query_test_set(fixture)


def test_load_query_test_set_rejects_duplicate_query_id(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        "L24_V035,642\n",
    )
    value = build_query_test_set(root, "002", ground_truth_source="human_verified")
    value["cases"].append(dict(value["cases"][0]))
    value["cases"][1]["source_query_file"] = "002/different.txt"
    fixture = tmp_path / "query_002.json"
    fixture.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate query_id"):
        query_test_set_module.load_query_test_set(fixture)


def test_load_query_test_set_rejects_invalid_source_sha256(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        "L24_V035,642\n",
    )
    value = build_query_test_set(root, "002", ground_truth_source="human_verified")
    value["source_sha256"] = "invalid_hash"
    fixture = tmp_path / "query_002.json"
    fixture.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="source_sha256 must be a 64-character lowercase hex string"):
        query_test_set_module.load_query_test_set(fixture)


def test_load_query_test_set_rejects_answer_on_non_qa(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write(root / "002" / "query-p2-1-kis.txt", "Một cảnh cần tìm.")
    _write(
        root / "002" / "ground_truth" / "query-p2-1-kis.csv",
        "L24_V035,642\n",
    )
    value = build_query_test_set(root, "002", ground_truth_source="human_verified")
    value["cases"][0]["answer"] = "illegal answer"
    fixture = tmp_path / "query_002.json"
    fixture.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="non-QA case must have null answer"):
        query_test_set_module.load_query_test_set(fixture)


def test_authoritative_fixture_query_002_integrity() -> None:
    fixture_path = Path("artifacts/evaluation/query_002.json")
    assert fixture_path.is_file(), f"Fixture file not found at {fixture_path}"

    test_set = query_test_set_module.load_query_test_set(fixture_path)

    assert test_set["schema_version"] == "hcmai-query-test-set-v1"
    assert test_set["split"] == "002"
    assert test_set["ground_truth_source"] == "human_verified"
    assert test_set["qa_mode"] == "retrieval_only"
    assert test_set["source_sha256"] == "52621db90283fd54c8b89a870f93e0704090cb0470f04fd77e90d161815e75cb"

    cases = test_set["cases"]
    assert len(cases) == 30

    kis_cases = [c for c in cases if c["kind"] == "kis"]
    qa_cases = [c for c in cases if c["kind"] == "qa"]
    trake_cases = [c for c in cases if c["kind"] == "trake"]

    assert len(kis_cases) == 19
    assert len(qa_cases) == 9
    assert len(trake_cases) == 2

    total_gt_rows = 0
    for case in cases:
        gt = case["ground_truth"]
        assert len(gt) == 100, f"{case['query_id']} has {len(gt)} ground truth rows, expected 100"
        total_gt_rows += len(gt)

        if case["kind"] == "qa":
            assert case["answer"] is not None and isinstance(case["answer"], str)
            assert case["raw_query"] != case["retrieval_query"]
            assert case["retrieval_query"] == " ".join(case["events"])
        elif case["kind"] == "kis":
            assert case["answer"] is None
            assert case["raw_query"] == case["retrieval_query"]
        elif case["kind"] == "trake":
            assert case["answer"] is None
            for entry in gt:
                assert len(entry["frame_idxs"]) == len(case["events"])

    assert total_gt_rows == 3000


def test_authoritative_fixture_query_002_windows_integrity() -> None:
    fixture_path = Path("artifacts/evaluation/query_002_windows.json")
    assert fixture_path.is_file(), f"Fixture file not found at {fixture_path}"

    test_set = query_test_set_module.load_query_test_set(fixture_path)

    assert test_set["schema_version"] == "hcmai-query-test-set-v2"
    assert test_set["split"] == "002"
    assert test_set["ground_truth_source"] == "human_audited_windows"
    assert test_set["qa_mode"] == "retrieval_only"

    cases = test_set["cases"]
    assert len(cases) == 30

    kis_cases = [c for c in cases if c["kind"] == "kis"]
    qa_cases = [c for c in cases if c["kind"] == "qa"]
    trake_cases = [c for c in cases if c["kind"] == "trake"]

    assert len(kis_cases) == 19
    assert len(qa_cases) == 9
    assert len(trake_cases) == 2

    for case in cases:
        gt = case["ground_truth"]
        assert len(gt) >= 1
        for entry in gt:
            assert "video_id" in entry
            assert "time_windows_ms" in entry or "frame_windows" in entry
            if "time_windows_ms" in entry:
                for start_ms, end_ms in entry["time_windows_ms"]:
                    assert 0 <= start_ms <= end_ms
            if "frame_windows" in entry:
                for start_f, end_f in entry["frame_windows"]:
                    assert 0 <= start_f <= end_f

        if case["kind"] == "qa":
            assert case["answer"] is not None and isinstance(case["answer"], str)
            assert case["retrieval_query"] == " ".join(case["events"])
        elif case["kind"] == "kis":
            assert case["answer"] is None
            assert case["raw_query"] == case["retrieval_query"]
        elif case["kind"] == "trake":
            assert case["answer"] is None
            for entry in gt:
                if "event_windows" in entry:
                    assert len(entry["event_windows"]) == len(case["events"])


def test_authoritative_fixture_query_001_windows_integrity() -> None:
    fixture_path = Path("artifacts/evaluation/query_001_windows.json")
    assert fixture_path.is_file(), f"Fixture file not found at {fixture_path}"

    test_set = query_test_set_module.load_query_test_set(fixture_path)

    assert test_set["schema_version"] == "hcmai-query-test-set-v2"
    assert test_set["split"] == "001"
    assert test_set["ground_truth_source"] == "human_audited_windows"
    assert test_set["qa_mode"] == "retrieval_only"

    cases = test_set["cases"]
    assert len(cases) == 25

    kis_cases = [c for c in cases if c["kind"] == "kis"]
    qa_cases = [c for c in cases if c["kind"] == "qa"]
    trake_cases = [c for c in cases if c["kind"] == "trake"]

    assert len(kis_cases) == 20
    assert len(qa_cases) == 4
    assert len(trake_cases) == 1

    for case in cases:
        gt = case["ground_truth"]
        assert len(gt) >= 1
        for entry in gt:
            assert "video_id" in entry
            assert "time_windows_ms" in entry or "frame_windows" in entry
            if "time_windows_ms" in entry:
                for start_ms, end_ms in entry["time_windows_ms"]:
                    assert 0 <= start_ms <= end_ms
            if "frame_windows" in entry:
                for start_f, end_f in entry["frame_windows"]:
                    assert 0 <= start_f <= end_f

        if case["kind"] == "qa":
            assert case["answer"] is not None and isinstance(case["answer"], str)
            assert case["retrieval_query"] == " ".join(case["events"])
        elif case["kind"] == "kis":
            assert case["answer"] is None
            assert case["raw_query"] == case["retrieval_query"]
        elif case["kind"] == "trake":
            assert case["answer"] is None
            for entry in gt:
                if "event_windows" in entry:
                    assert len(entry["event_windows"]) == len(case["events"])


def test_authoritative_fixture_query_003_windows_integrity() -> None:
    fixture_path = Path("artifacts/evaluation/query_003_windows.json")
    assert fixture_path.is_file(), f"Fixture file not found at {fixture_path}"

    test_set = query_test_set_module.load_query_test_set(fixture_path)

    assert test_set["schema_version"] == "hcmai-query-test-set-v2"
    assert test_set["split"] == "003"
    assert test_set["ground_truth_source"] == "human_audited_windows"
    assert test_set["qa_mode"] == "retrieval_only"

    cases = test_set["cases"]
    assert len(cases) == 36

    kis_cases = [c for c in cases if c["kind"] == "kis"]
    qa_cases = [c for c in cases if c["kind"] == "qa"]
    trake_cases = [c for c in cases if c["kind"] == "trake"]

    assert len(kis_cases) == 26
    assert len(qa_cases) == 8
    assert len(trake_cases) == 2

    for case in cases:
        gt = case["ground_truth"]
        assert len(gt) >= 1
        for entry in gt:
            assert "video_id" in entry
            assert "time_windows_ms" in entry or "frame_windows" in entry
            if "time_windows_ms" in entry:
                for start_ms, end_ms in entry["time_windows_ms"]:
                    assert 0 <= start_ms <= end_ms
            if "frame_windows" in entry:
                for start_f, end_f in entry["frame_windows"]:
                    assert 0 <= start_f <= end_f

        if case["kind"] == "qa":
            assert case["answer"] is not None and isinstance(case["answer"], str)
            assert case["retrieval_query"] == " ".join(case["events"])
        elif case["kind"] == "kis":
            assert case["answer"] is None
            assert case["raw_query"] == case["retrieval_query"]
        elif case["kind"] == "trake":
            assert case["answer"] is None
            for entry in gt:
                if "event_windows" in entry:
                    assert len(entry["event_windows"]) == len(case["events"])


