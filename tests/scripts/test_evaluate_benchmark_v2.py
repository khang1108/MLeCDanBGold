"""Compatibility tests for the identity- and time-aware benchmark evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.evaluation.evaluate_benchmark_v2 import evaluate, load_responses


def _metadata(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 30,
                "timestamp_ms": 1_000,
                "fps": 30.0,
            }
        ]
    ).to_parquet(path, index=False)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_responses_accepts_historical_list_and_versioned_envelope(tmp_path: Path) -> None:
    responses = [{"query_file": "q-kis.txt", "status_code": 200}]
    historical = tmp_path / "historical.json"
    envelope = tmp_path / "envelope.json"
    _write_json(historical, responses)
    _write_json(envelope, {"schema_version": "hcmai-baseline-run-v1", "responses": responses})

    assert load_responses(historical) == responses
    assert load_responses(envelope) == responses


@pytest.mark.parametrize("value", [{}, {"responses": {}}, "invalid"])
def test_load_responses_rejects_invalid_envelopes(tmp_path: Path, value: object) -> None:
    path = tmp_path / "invalid.json"
    _write_json(path, value)

    with pytest.raises(ValueError, match="responses"):
        load_responses(path)


def test_equivalent_list_and_envelope_produce_identical_metrics(tmp_path: Path) -> None:
    metadata = tmp_path / "frames.parquet"
    _metadata(metadata)
    query_root = tmp_path / "queries"
    ground_truth = query_root / "002" / "ground_truth"
    ground_truth.mkdir(parents=True)
    (ground_truth / "q-kis.csv").write_text("v1,30\n", encoding="utf-8")
    responses = [
        {
            "query_file": str(query_root / "002" / "q-kis.txt"),
            "type": "kis",
            "status_code": 200,
            "elapsed_wall_s": 0.1,
            "response": {
                "results": [
                    {
                        "video_id": "v1",
                        "frame_id": "f1",
                        "frame_ids": ["f1"],
                    }
                ],
                "latency": {"total_ms": 10.0, "retrieval_ms": 8.0, "alignment_ms": 2.0},
            },
        }
    ]
    historical = tmp_path / "historical.json"
    envelope = tmp_path / "envelope.json"
    _write_json(historical, responses)
    _write_json(envelope, {"schema_version": "hcmai-baseline-run-v1", "responses": responses})

    old = evaluate(historical, query_root, metadata, 0.0)
    new = evaluate(envelope, query_root, metadata, 0.0)

    old.pop("response_file")
    new.pop("response_file")
    assert new == old


def test_global_trake_candidate_counts_for_video_but_not_event_grounding(tmp_path: Path) -> None:
    metadata = tmp_path / "frames.parquet"
    _metadata(metadata)
    query_root = tmp_path / "queries"
    ground_truth = query_root / "002" / "ground_truth"
    ground_truth.mkdir(parents=True)
    (ground_truth / "q-trake.csv").write_text("v1,30,60,90,120\n", encoding="utf-8")
    response_path = tmp_path / "run.json"
    _write_json(
        response_path,
        {
            "responses": [
                {
                    "query_file": str(query_root / "002" / "q-trake.txt"),
                    "type": "trake",
                    "status_code": 200,
                    "elapsed_wall_s": 0.1,
                    "response": {
                        "events": ["a", "b", "c", "d"],
                        "paths": [
                            {
                                "video_id": "v1",
                                "frame_ids": ["f1"],
                                "frame_idxs": [30],
                            }
                        ],
                        "latency": {
                            "total_ms": 10.0,
                            "retrieval_ms": 8.0,
                            "alignment_ms": 2.0,
                        },
                    },
                }
            ]
        },
    )

    result = evaluate(response_path, query_root, metadata, 0.0)
    row = result["per_query"][0]

    assert row["video_first_rank"] == 1
    assert row["representative_first_rank"] is None
    assert row["path_first_rank"] is None
    assert row["event_matrix"] == [[False, False, False, False]]
    assert result["trake"]["video_recall@1"] == 1.0
    assert result["trake"]["event_recall@1"] == 0.0


def test_evaluate_with_fixture_kis_qa_trake(tmp_path: Path) -> None:
    metadata = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            {"frame_id": "f1", "video_id": "v1", "frame_idx": 30, "timestamp_ms": 1000, "fps": 30.0},
            {"frame_id": "f2", "video_id": "v1", "frame_idx": 60, "timestamp_ms": 2000, "fps": 30.0},
            {"frame_id": "f3", "video_id": "v2", "frame_idx": 100, "timestamp_ms": 3333, "fps": 30.0},
        ]
    ).to_parquet(metadata, index=False)

    fixture_data = {
        "schema_version": "hcmai-query-test-set-v1",
        "split": "002",
        "ground_truth_source": "human_verified",
        "qa_mode": "retrieval_only",
        "source_sha256": "0" * 64,
        "cases": [
            {
                "query_id": "q-kis",
                "kind": "kis",
                "source_query_file": "002/q-kis.txt",
                "raw_query": "Cảnh KIS",
                "retrieval_query": "Cảnh KIS",
                "events": ["Cảnh KIS"],
                "answer": None,
                "ground_truth": [{"video_id": "v1", "frame_idxs": [30]}],
            },
            {
                "query_id": "q-qa",
                "kind": "qa",
                "source_query_file": "002/q-qa.txt",
                "raw_query": "Cảnh QA. Đây là con cá gì?",
                "retrieval_query": "Cảnh QA",
                "events": ["Cảnh QA"],
                "answer": "Cá Bống",
                "ground_truth": [{"video_id": "v2", "frame_idxs": [100]}],
            },
            {
                "query_id": "q-trake",
                "kind": "trake",
                "source_query_file": "002/q-trake.txt",
                "raw_query": "E1: Sự kiện 1.\nE2: Sự kiện 2.",
                "retrieval_query": "Sự kiện 1. Sự kiện 2.",
                "events": ["Sự kiện 1.", "Sự kiện 2."],
                "answer": None,
                "ground_truth": [{"video_id": "v1", "frame_idxs": [30, 60]}],
            },
        ],
    }
    fixture_path = tmp_path / "test_set.json"
    _write_json(fixture_path, fixture_data)

    response_data = [
        {
            "query_id": "q-kis",
            "query_file": "002/q-kis.txt",
            "type": "kis",
            "status_code": 200,
            "elapsed_wall_s": 0.1,
            "response": {
                "results": [{"video_id": "v1", "frame_id": "f1", "frame_ids": ["f1"]}],
                "latency": {"total_ms": 10.0, "retrieval_ms": 8.0, "alignment_ms": 2.0},
            },
        },
        {
            "query_id": "q-qa",
            "query_file": "002/q-qa.txt",
            "type": "qa",
            "status_code": 200,
            "elapsed_wall_s": 0.12,
            "response": {
                "results": [{"video_id": "v2", "frame_id": "f3", "frame_ids": ["f3"]}],
                "latency": {"total_ms": 12.0, "retrieval_ms": 9.0, "alignment_ms": 3.0},
            },
        },
        {
            "query_id": "q-trake",
            "query_file": "002/q-trake.txt",
            "type": "trake",
            "status_code": 200,
            "elapsed_wall_s": 0.25,
            "response": {
                "paths": [{"video_id": "v1", "frame_idxs": [30, 60]}],
                "latency": {"total_ms": 25.0, "retrieval_ms": 15.0, "alignment_ms": 10.0},
            },
        },
    ]
    response_path = tmp_path / "responses.json"
    _write_json(response_path, response_data)

    result = evaluate(response_path, None, metadata, 0.0, test_set_path=fixture_path)

    assert result["evaluated_queries"] == 3
    assert len(result["skipped"]) == 0

    # Verify KIS metrics
    assert result["kis"]["video_recall@1"] == 1.0
    assert result["kis"]["representative_recall@1"] == 1.0
    assert result["kis"]["representative_mrr"] == 1.0

    # Verify QA metrics are calculated as retrieval-only
    assert result["qa"]["video_recall@1"] == 1.0
    assert result["qa"]["representative_recall@1"] == 1.0
    assert result["qa"]["representative_mrr"] == 1.0

    # Verify TRAKE metrics
    assert result["trake"]["video_recall@1"] == 1.0
    assert result["trake"]["representative_recall@1"] == 1.0  # allHit
    assert result["trake"]["event_recall@1"] == 1.0


def test_evaluate_fixture_enforces_authoritative_trake_event_count(tmp_path: Path) -> None:
    metadata = tmp_path / "frames.parquet"
    _metadata(metadata)

    fixture_data = {
        "schema_version": "hcmai-query-test-set-v1",
        "split": "002",
        "ground_truth_source": "human_verified",
        "qa_mode": "retrieval_only",
        "source_sha256": "0" * 64,
        "cases": [
            {
                "query_id": "q-trake",
                "kind": "trake",
                "source_query_file": "002/q-trake.txt",
                "raw_query": "E1: A.\nE2: B.\nE3: C.",
                "retrieval_query": "A. B. C.",
                "events": ["A.", "B.", "C."],
                "answer": None,
                "ground_truth": [{"video_id": "v1", "frame_idxs": [30, 60, 90]}],
            },
        ],
    }
    fixture_path = tmp_path / "test_set.json"
    _write_json(fixture_path, fixture_data)

    # Response falsely claims 2 events and gives 2 frames
    response_data = [
        {
            "query_file": "002/q-trake.txt",
            "type": "trake",
            "status_code": 200,
            "elapsed_wall_s": 0.1,
            "response": {
                "events": ["A.", "B."],  # Only 2 events in response
                "paths": [{"video_id": "v1", "frame_idxs": [30, 60]}],
            },
        }
    ]
    response_path = tmp_path / "responses.json"
    _write_json(response_path, response_data)

    result = evaluate(response_path, None, metadata, 0.0, test_set_path=fixture_path)
    row = result["per_query"][0]

    # Because fixture requires 3 events, candidate with 2 frames fails path matching
    assert row["representative_first_rank"] is None
    assert row["path_first_rank"] is None
    assert row["event_matrix"] == [[False, False, False]]
    assert result["trake"]["event_recall@1"] == 0.0


def test_evaluate_fixture_handles_failed_and_unlabeled_queries(tmp_path: Path) -> None:
    metadata = tmp_path / "frames.parquet"
    _metadata(metadata)

    fixture_data = {
        "schema_version": "hcmai-query-test-set-v1",
        "split": "002",
        "ground_truth_source": "human_verified",
        "qa_mode": "retrieval_only",
        "source_sha256": "0" * 64,
        "cases": [
            {
                "query_id": "q-kis",
                "kind": "kis",
                "source_query_file": "002/q-kis.txt",
                "raw_query": "Cảnh KIS",
                "retrieval_query": "Cảnh KIS",
                "events": ["Cảnh KIS"],
                "answer": None,
                "ground_truth": [{"video_id": "v1", "frame_idxs": [30]}],
            },
        ],
    }
    fixture_path = tmp_path / "test_set.json"
    _write_json(fixture_path, fixture_data)

    response_data = [
        {
            "query_id": "q-kis",
            "query_file": "002/q-kis.txt",
            "type": "kis",
            "status_code": 500,  # Failed request
            "response": {"raw": "Internal Server Error"},
        },
        {
            "query_id": "q-unknown",
            "query_file": "002/q-unknown.txt",
            "type": "kis",
            "status_code": 200,  # Unknown query not in fixture
            "response": {"results": []},
        },
    ]
    response_path = tmp_path / "responses.json"
    _write_json(response_path, response_data)

    result = evaluate(response_path, None, metadata, 0.0, test_set_path=fixture_path)

    assert result["evaluated_queries"] == 0
    assert len(result["skipped"]) == 2
    assert result["skipped"][0]["reason"] == "request_failed"
    assert result["skipped"][1]["reason"] == "no_ground_truth"


def test_evaluate_fixture_supports_exact_and_relaxed_matching(tmp_path: Path) -> None:
    metadata = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            # At 30 fps, frame 30 is 1000ms, frame 45 is 1500ms
            {"frame_id": "f_close", "video_id": "v1", "frame_idx": 45, "timestamp_ms": 1500, "fps": 30.0},
        ]
    ).to_parquet(metadata, index=False)

    fixture_data = {
        "schema_version": "hcmai-query-test-set-v1",
        "split": "002",
        "ground_truth_source": "human_verified",
        "qa_mode": "retrieval_only",
        "source_sha256": "0" * 64,
        "cases": [
            {
                "query_id": "q-kis",
                "kind": "kis",
                "source_query_file": "002/q-kis.txt",
                "raw_query": "Cảnh KIS",
                "retrieval_query": "Cảnh KIS",
                "events": ["Cảnh KIS"],
                "answer": None,
                "ground_truth": [{"video_id": "v1", "frame_idxs": [30]}],  # ground truth is frame 30
            },
        ],
    }
    fixture_path = tmp_path / "test_set.json"
    _write_json(fixture_path, fixture_data)

    response_data = [
        {
            "query_id": "q-kis",
            "query_file": "002/q-kis.txt",
            "type": "kis",
            "status_code": 200,
            "response": {
                "results": [{"video_id": "v1", "frame_id": "f_close", "frame_ids": ["f_close"]}],
            },
        }
    ]
    response_path = tmp_path / "responses.json"
    _write_json(response_path, response_data)

    # Exact (tolerance = 0.0s): frame 45 != frame 30 -> representative hit False
    exact = evaluate(response_path, None, metadata, 0.0, test_set_path=fixture_path)
    assert exact["kis"]["representative_recall@1"] == 0.0

    # Relaxed (tolerance = 1.0s): |1500 - 1000| = 500ms <= 1000ms -> representative hit True
    relaxed = evaluate(response_path, None, metadata, 1.0, test_set_path=fixture_path)
    assert relaxed["kis"]["representative_recall@1"] == 1.0


def test_evaluate_with_window_ground_truth(tmp_path: Path) -> None:
    metadata = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            {"frame_id": "f_in", "video_id": "v1", "frame_idx": 50, "timestamp_ms": 2_000, "fps": 25.0},
            {"frame_id": "f_out", "video_id": "v1", "frame_idx": 150, "timestamp_ms": 6_000, "fps": 25.0},
        ]
    ).to_parquet(metadata, index=False)

    fixture_data = {
        "schema_version": "hcmai-query-test-set-v2",
        "split": "002",
        "ground_truth_source": "human_audited_windows",
        "qa_mode": "retrieval_only",
        "source_sha256": "52621db90283fd54c8b89a870f93e0704090cb0470f04fd77e90d161815e75cb",
        "cases": [
            {
                "query_id": "q-win-kis",
                "kind": "kis",
                "source_query_file": "002/q-win-kis.txt",
                "raw_query": "Cảnh cửa sổ",
                "retrieval_query": "Cảnh cửa sổ",
                "events": ["Cảnh cửa sổ"],
                "answer": None,
                "ground_truth": [
                    {
                        "video_id": "v1",
                        "time_windows_ms": [[1000, 3000]],
                        "frame_windows": [[25, 75]],
                        "representative_frame_idx": 50,
                        "frame_idxs": [50],
                    }
                ],
            }
        ],
    }
    fixture_path = tmp_path / "test_set_windows.json"
    _write_json(fixture_path, fixture_data)

    # Response with candidate inside window
    resp_in = [
        {
            "query_id": "q-win-kis",
            "type": "kis",
            "status_code": 200,
            "response": {
                "results": [{"video_id": "v1", "frame_id": "f_in", "frame_ids": ["f_in"]}],
            },
        }
    ]
    resp_in_path = tmp_path / "resp_in.json"
    _write_json(resp_in_path, resp_in)
    res_in = evaluate(resp_in_path, None, metadata, 0.0, test_set_path=fixture_path)
    assert res_in["kis"]["representative_recall@1"] == 1.0

    # Response with candidate outside window
    resp_out = [
        {
            "query_id": "q-win-kis",
            "type": "kis",
            "status_code": 200,
            "response": {
                "results": [{"video_id": "v1", "frame_id": "f_out", "frame_ids": ["f_out"]}],
            },
        }
    ]
    resp_out_path = tmp_path / "resp_out.json"
    _write_json(resp_out_path, resp_out)
    res_out = evaluate(resp_out_path, None, metadata, 0.0, test_set_path=fixture_path)
    assert res_out["kis"]["representative_recall@1"] == 0.0

