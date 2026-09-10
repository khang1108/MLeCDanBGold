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
