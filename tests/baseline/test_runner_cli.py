"""Tests for reproducible baseline execution and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from baseline.cli import main, parse_args
from baseline.contracts import MethodOptions, QueryCase
from baseline.runner import RunOptions, run_cases, write_run
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class _Metadata:
    def to_dict(self) -> dict[str, object]:
        return {"dataset_version": "fixture-v1", "model_name": "fixture-encoder"}


class _Evidence:
    def __init__(self) -> None:
        self.dense = object()
        self.bm25 = object()
        self.visual_index = SimpleNamespace(metadata=_Metadata())
        self.calls: list[dict[str, object]] = []

    def score_events(
        self,
        original_events: tuple[str, ...],
        retrieval_events: tuple[str, ...],
        *,
        caption_events: tuple[str, ...] | None,
        use_dense: bool,
        use_bm25: bool,
    ) -> list[VideoEventScores]:
        self.calls.append(
            {
                "original_events": original_events,
                "retrieval_events": retrieval_events,
                "caption_events": caption_events,
                "use_dense": use_dense,
                "use_bm25": use_bm25,
            }
        )
        event_count = len(original_events)
        return [_video("v1", [[0.2, 0.8] for _ in range(event_count)])]


class _FailingEvidence(_Evidence):
    def score_events(
        self,
        original_events: tuple[str, ...],
        retrieval_events: tuple[str, ...],
        *,
        caption_events: tuple[str, ...] | None,
        use_dense: bool,
        use_bm25: bool,
    ) -> list[VideoEventScores]:
        if original_events == ("fail",):
            raise RuntimeError("fixture failure")
        return super().score_events(
            original_events,
            retrieval_events,
            caption_events=caption_events,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )


class _Service:
    def __init__(self, evidence: _Evidence | None = None) -> None:
        self.corpus = object()
        self.temporal_evidence = evidence or _Evidence()
        self.config = SimpleNamespace(
            alignment=SimpleNamespace(
                lambda_gap=0.00001,
                event_power=1.0,
                cluster_delta=0.0,
            )
        )
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _video(video_id: str, scores: list[list[float]]) -> VideoEventScores:
    matrix = np.asarray(scores, dtype=np.float32)
    frame_count = matrix.shape[1]
    return VideoEventScores(
        video_id=video_id,
        frame_ids=np.asarray([f"{video_id}-f{i}" for i in range(frame_count)]),
        frame_idx=np.arange(frame_count, dtype=np.int64),
        timestamps_ms=np.arange(frame_count, dtype=np.int64) * 1_000,
        scores=matrix,
    )


def _options(method: str, *, use_bm25: bool = False) -> RunOptions:
    return RunOptions(
        method=method,  # type: ignore[arg-type]
        top_k=10,
        use_dense=True,
        use_bm25=use_bm25,
        method_options=MethodOptions(lambda_gap=0.0),
    )


def test_global_runner_scores_retrieval_query_but_preserves_raw_query_and_events() -> None:
    service = _Service()
    case = QueryCase("q-trake.txt", "trake", "header\nE1: alpha\nE2: beta", ("alpha", "beta"))

    run = run_cases(service, [case], _options("global_query"))

    assert service.temporal_evidence.calls == [
        {
            "original_events": (case.retrieval_query,),
            "retrieval_events": (case.retrieval_query,),
            "caption_events": None,
            "use_dense": True,
            "use_bm25": False,
        }
    ]
    item = run["responses"][0]  # type: ignore[index]
    assert item["status_code"] == 200
    assert item["response"]["raw_query"] == case.raw_query
    assert item["response"]["retrieval_query"] == "alpha beta"
    assert item["response"]["events"] == ["alpha", "beta"]
    assert len(item["response"]["paths"][0]["frame_ids"]) == 1


def test_event_runner_scores_events_and_supplies_bm25_caption_text() -> None:
    service = _Service()
    case = QueryCase("q-kis.txt", "kis", "alpha\nbeta", ("alpha", "beta"))

    run = run_cases(service, [case], _options("independent_events", use_bm25=True))

    assert service.temporal_evidence.calls[0]["original_events"] == ("alpha", "beta")
    assert service.temporal_evidence.calls[0]["retrieval_events"] == ("alpha", "beta")
    assert service.temporal_evidence.calls[0]["caption_events"] == ("alpha", "beta")
    candidate = run["responses"][0]["response"]["results"][0]  # type: ignore[index]
    assert candidate["frame_id"] == "v1-f1"
    assert candidate["frame_ids"] == ["v1-f1", "v1-f1"]


def test_runner_isolates_query_failure_and_records_provenance_and_timings() -> None:
    service = _Service(_FailingEvidence())
    cases = [
        QueryCase("ok-kis.txt", "kis", "ok", ("ok",)),
        QueryCase("fail-kis.txt", "kis", "fail", ("fail",)),
    ]

    run = run_cases(service, cases, _options("global_query"))

    assert run["schema_version"] == "hcmai-baseline-run-v1"
    assert run["run"]["index_metadata"]["dataset_version"] == "fixture-v1"  # type: ignore[index]
    responses = cast(list[dict[str, Any]], run["responses"])
    assert [item["status_code"] for item in responses] == [200, 0]
    assert responses[1]["error"] == "RuntimeError: fixture failure"
    latency = responses[0]["response"]["latency"]
    assert set(latency) >= {"retrieval_ms", "alignment_ms", "total_ms"}
    assert responses[0]["elapsed_wall_s"] >= 0


def test_write_run_uses_valid_json_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "run.json"
    value = {"schema_version": "hcmai-baseline-run-v1", "run": {}, "responses": []}

    write_run(output, value)

    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert list(output.parent.glob("*.tmp")) == []
    assert list(output.parent.glob(".*.tmp")) == []


def test_runner_rejects_missing_requested_capability() -> None:
    service = _Service()
    service.temporal_evidence.dense = None

    with pytest.raises(RuntimeError, match="Dense evidence"):
        run_cases(
            service,
            [QueryCase("q-kis.txt", "kis", "query", ("query",))],
            _options("global_query"),
        )


@pytest.mark.parametrize(
    ("method", "invalid_args"),
    [
        ("global_query", ["--lambda-gap", "0"]),
        ("unary_candidate_lattice", ["--candidates-per-event", "0"]),
        ("unary_bounded_order", ["--max-order-candidates", "0"]),
    ],
)
def test_parse_args_rejects_irrelevant_or_invalid_flags(
    method: str,
    invalid_args: list[str],
) -> None:
    with pytest.raises(SystemExit):
        parse_args(method, ["--output", "run.json", *invalid_args])  # type: ignore[arg-type]


def test_parse_args_rejects_invalid_common_controls() -> None:
    with pytest.raises(SystemExit):
        parse_args("global_query", ["--output", "run.json", "--top-k", "0"])
    with pytest.raises(SystemExit):
        parse_args(
            "global_query",
            ["--output", "run.json", "--no-use-dense", "--no-use-bm25"],
        )


def test_cli_loads_service_once_writes_envelope_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_root = tmp_path / "queries"
    query_path = query_root / "002" / "one-kis.txt"
    query_path.parent.mkdir(parents=True)
    query_path.write_text("one query", encoding="utf-8")
    output = tmp_path / "run.json"
    service = _Service()
    loads: list[list[str]] = []

    def load_service(messages: list[str]) -> _Service:
        loads.append(messages)
        return service

    monkeypatch.setattr("baseline.cli._load_service", load_service)

    result = main(
        "global_query",
        [
            "--query-root",
            str(query_root),
            "--splits",
            "002",
            "--kinds",
            "kis",
            "--output",
            str(output),
        ],
    )

    assert result == 0
    assert len(loads) == 1
    assert service.closed is True
    run = json.loads(output.read_text(encoding="utf-8"))
    assert run["run"]["method"] == "global_query"
    assert run["run"]["use_bm25"] is False
    assert len(run["responses"]) == 1


def test_cli_returns_failure_after_writing_failed_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_root = tmp_path / "queries"
    query_path = query_root / "002" / "one-kis.txt"
    query_path.parent.mkdir(parents=True)
    query_path.write_text("fail", encoding="utf-8")
    output = tmp_path / "run.json"
    service = _Service(_FailingEvidence())
    monkeypatch.setattr("baseline.cli._load_service", lambda _messages: service)

    result = main(
        "global_query",
        [
            "--query-root",
            str(query_root),
            "--splits",
            "002",
            "--kinds",
            "kis",
            "--output",
            str(output),
        ],
    )

    assert result == 1
    assert service.closed is True
    run = json.loads(output.read_text(encoding="utf-8"))
    assert run["responses"][0]["status_code"] == 0
