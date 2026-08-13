"""Fallback event/question retrieval branches and frame merge."""

from __future__ import annotations

from hcmai.common.schemas import (
    FrameEvidence,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    TaskType,
)
from hcmai.common.schemas.search import SearchFilters
from ..domain.models import ParsedVQAQuery
from ..domain.ports import FrameLookup, RetrievalGateway


def retrieve_candidates(
    retrieval: RetrievalGateway,
    data: FrameLookup,
    parsed: ParsedVQAQuery,
    *,
    top_k: int = 100,
    filters: SearchFilters | None = None,
    event_only: bool = False,
    question_only: bool = False,
) -> tuple[list[FrameEvidence], list[str]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if event_only and question_only:
        raise ValueError("event_only and question_only are mutually exclusive")
    branches: list[tuple[str, str]] = []
    if not question_only:
        branches.append(("event", parsed.retrieval_query))
    if not event_only:
        branches.append(("question", parsed.question))
        branches.extend(("clue", clue) for clue in parsed.clue_queries if clue != parsed.question)
    results = retrieval.search_batch(
        [query for _, query in branches], top_k, filters, TaskType.VQA
    )
    warnings = [warning for result in results for warning in result.warnings]
    return _merge(branches, results, data, parsed), warnings


def _merge(
    branches: list[tuple[str, str]],
    results: list[RetrievalResult],
    data: FrameLookup,
    parsed: ParsedVQAQuery,
) -> list[FrameEvidence]:
    by_id: dict[str, dict[str, object]] = {}
    for (branch, _), result in zip(branches, results, strict=True):
        for rank, candidate in enumerate(result.candidates, 1):
            entry = by_id.setdefault(candidate.frame_id, {"branches": {}, "candidates": []})
            score = _candidate_score(candidate, rank)
            branch_weight = 1.0 if branch == "event" else 0.9
            entry["branches"][branch] = max(  # type: ignore[index]
                entry["branches"].get(branch, 0.0), score * branch_weight  # type: ignore[union-attr]
            )
            entry["candidates"].append(candidate)  # type: ignore[union-attr]
    merged: list[FrameEvidence] = []
    for frame_id, entry in by_id.items():
        candidates = entry["candidates"]
        unit_scores = entry["branches"]
        source_scores: dict[RetrievalSource, float] = {}
        source_ranks: dict[RetrievalSource, int] = {}
        for candidate in candidates:  # type: ignore[union-attr]
            for source, score in candidate.source_scores.items():
                boost = 1.15 if source in parsed.required_modalities else 1.0
                source_scores[source] = max(source_scores.get(source, float("-inf")), score * boost)
            for source, rank in candidate.source_ranks.items():
                current = source_ranks.get(source)
                source_ranks[source] = rank if current is None else min(current, rank)
        score = max(unit_scores.values()) + 0.15 * sum(unit_scores.values())  # type: ignore[union-attr]
        merged.append(FrameEvidence(
            frame=data.get_frame(frame_id),
            unit_scores=dict(unit_scores),  # type: ignore[arg-type]
            source_scores=source_scores,
            source_ranks=source_ranks,
            score=score,
            provenance=tuple(sorted(unit_scores)),  # type: ignore[arg-type]
        ))
    return sorted(merged, key=lambda item: (-item.score, item.frame.video_id, item.frame.frame_idx, item.frame.frame_id))


def _candidate_score(candidate: RetrievalCandidate, rank: int) -> float:
    for value in (candidate.final_score, candidate.reranker_score, candidate.fusion_score):
        if value is not None:
            return float(value)
    return 1.0 / rank
