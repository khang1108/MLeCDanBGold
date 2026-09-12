"""Execute baseline methods over shared full-corpus temporal evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any

from baseline.contracts import BASELINE_METHODS, BaselinePath, MethodName, MethodOptions, QueryCase
from baseline.methods import create_method


RUN_SCHEMA_VERSION = "hcmai-baseline-run-v1"


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Method and evidence settings fixed for one output artifact."""

    method: MethodName
    top_k: int
    use_dense: bool
    use_bm25: bool
    method_options: MethodOptions

    def __post_init__(self) -> None:
        if self.method not in BASELINE_METHODS:
            raise ValueError(f"unsupported baseline method: {self.method}")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not self.use_dense and not self.use_bm25:
            raise ValueError("at least one evidence source must be enabled")


def run_cases(
    service: Any,
    cases: Sequence[QueryCase],
    options: RunOptions,
) -> dict[str, object]:
    """Score each query once and isolate failures in an evaluator-compatible run."""

    evidence = _validate_service(service, options)
    method = create_method(options.method, options.method_options)
    responses: list[dict[str, object]] = []
    for case in cases:
        started = perf_counter()
        try:
            scoring_events = (
                (case.retrieval_query,)
                if options.method == "global_query"
                else case.events
            )
            caption_events = scoring_events if options.use_bm25 else None
            retrieval_started = perf_counter()
            videos = evidence.score_events(
                scoring_events,
                scoring_events,
                caption_events=caption_events,
                use_dense=options.use_dense,
                use_bm25=options.use_bm25,
            )
            retrieval_ms = (perf_counter() - retrieval_started) * 1_000

            alignment_started = perf_counter()
            paths = method.rank(videos, top_k=options.top_k)
            alignment_ms = (perf_counter() - alignment_started) * 1_000
            total_ms = (perf_counter() - started) * 1_000
            response = _response_body(
                case,
                scoring_events,
                paths,
                options,
                retrieval_ms=retrieval_ms,
                alignment_ms=alignment_ms,
                total_ms=total_ms,
            )
            responses.append(
                {
                    "query_file": case.query_file,
                    "type": case.kind,
                    "status_code": 200,
                    "elapsed_wall_s": total_ms / 1_000,
                    "response": response,
                }
            )
        except Exception as error:
            responses.append(
                {
                    "query_file": case.query_file,
                    "type": case.kind,
                    "status_code": 0,
                    "elapsed_wall_s": perf_counter() - started,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run": {
            "created_at": datetime.now(UTC).isoformat(),
            "method": options.method,
            "top_k": options.top_k,
            "use_dense": options.use_dense,
            "use_bm25": options.use_bm25,
            "query_count": len(cases),
            "method_options": asdict(options.method_options),
            "index_metadata": evidence.visual_index.metadata.to_dict(),
        },
        "responses": responses,
    }


def write_run(path: Path, run: Mapping[str, object]) -> None:
    """Atomically publish one UTF-8 JSON run beside a temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(run, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _validate_service(service: Any, options: RunOptions) -> Any:
    if getattr(service, "corpus", None) is None:
        raise RuntimeError("Canonical corpus is unavailable")
    evidence = getattr(service, "temporal_evidence", None)
    if evidence is None:
        raise RuntimeError("Temporal evidence is unavailable")
    if options.use_dense and getattr(evidence, "dense", None) is None:
        raise RuntimeError("Dense evidence is unavailable")
    if options.use_bm25 and getattr(evidence, "bm25", None) is None:
        raise RuntimeError("BM25 evidence is unavailable")
    visual_index = getattr(evidence, "visual_index", None)
    metadata = getattr(visual_index, "metadata", None)
    if metadata is None or not callable(getattr(metadata, "to_dict", None)):
        raise RuntimeError("Visual index provenance is unavailable")
    return evidence


def _response_body(
    case: QueryCase,
    scoring_events: tuple[str, ...],
    paths: Sequence[BaselinePath],
    options: RunOptions,
    *,
    retrieval_ms: float,
    alignment_ms: float,
    total_ms: float,
) -> dict[str, object]:
    candidates = [_serialize_path(path, representative=case.kind != "trake") for path in paths]
    body: dict[str, object] = {
        "raw_query": case.raw_query,
        "retrieval_query": case.retrieval_query,
        "events": list(case.events),
        "scoring_events": list(scoring_events),
        "dense_events": list(scoring_events) if options.use_dense else None,
        "bm25_caption_events": list(scoring_events) if options.use_bm25 else None,
        "use_dense": options.use_dense,
        "use_bm25": options.use_bm25,
        "method": options.method,
        "latency": {
            "query_ms": 0.0,
            "retrieval_ms": retrieval_ms,
            "alignment_ms": alignment_ms,
            "materialization_ms": 0.0,
            "total_ms": total_ms,
        },
    }
    if case.kind == "trake":
        body["paths"] = candidates
    else:
        body["query"] = case.retrieval_query
        body["results"] = candidates
    return body


def _serialize_path(path: BaselinePath, *, representative: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "video_id": path.video_id,
        "score": path.score,
        "frame_ids": list(path.frame_ids),
        "frame_idxs": list(path.frame_idxs),
        "timestamps_ms": list(path.timestamps_ms),
        "chronological_event_order": (
            list(path.chronological_event_order)
            if path.chronological_event_order is not None
            else None
        ),
    }
    if representative:
        middle = len(path.frame_ids) // 2
        value.update(
            {
                "frame_id": path.frame_ids[middle],
                "frame_idx": path.frame_idxs[middle],
                "timestamp_ms": path.timestamps_ms[middle],
            }
        )
    return value
