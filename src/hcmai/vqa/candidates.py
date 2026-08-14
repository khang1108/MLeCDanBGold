"""Event/question retrieval branches and identity-preserving frame merge."""

from __future__ import annotations

from hcmai.common.schemas import (
    FrameLookup,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    TaskType,
)
from hcmai.common.schemas.search import SearchFilters
from .contracts import RetrievalGateway
from .models import BranchCandidate, ParsedVQAQuery


def retrieve_candidates(
    retrieval: RetrievalGateway,
    data: FrameLookup,
    parsed: ParsedVQAQuery,
    *,
    top_k: int = 100,
    filters: SearchFilters | None = None,
    event_only: bool = False,
    question_only: bool = False,
    modality_boost: float = 1.15,
) -> tuple[list[BranchCandidate], list[str]]:
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
    if modality_boost < 1.0:
        raise ValueError("modality_boost must be at least 1")
    results = retrieval.search_batch(
        [query for _, query in branches], top_k, filters, TaskType.VQA
    )
    warnings = [warning for result in results for warning in result.warnings]
    return _merge(branches, results, data, parsed, modality_boost), warnings


def _merge(
    branches: list[tuple[str, str]],
    results: list[RetrievalResult],
    data: FrameLookup,
    parsed: ParsedVQAQuery,
    modality_boost: float,
) -> list[BranchCandidate]:
    branches_by_id: dict[str, dict[str, float]] = {}
    candidates_by_id: dict[str, list[RetrievalCandidate]] = {}
    for (branch, _), result in zip(branches, results, strict=True):
        for rank, candidate in enumerate(result.candidates, 1):
            branch_scores = branches_by_id.setdefault(candidate.frame_id, {})
            score = _candidate_score(candidate, rank) * (1.0 if branch == "event" else 0.9)
            branch_scores[branch] = max(branch_scores.get(branch, 0.0), score)
            candidates_by_id.setdefault(candidate.frame_id, []).append(candidate)
    primary = next(iter(parsed.required_modalities), None)
    merged: list[BranchCandidate] = []
    for frame_id, branch_scores in branches_by_id.items():
        source_scores: dict[RetrievalSource, float] = {}
        source_ranks: dict[RetrievalSource, int] = {}
        for candidate in candidates_by_id[frame_id]:
            for source, score in candidate.source_scores.items():
                source_scores[source] = max(source_scores.get(source, float("-inf")), score)
            for source, rank in candidate.source_ranks.items():
                current = source_ranks.get(source)
                source_ranks[source] = rank if current is None else min(current, rank)
        boost = modality_boost if primary in source_scores else 1.0
        merged.append(BranchCandidate(
            frame=data.get_frame(frame_id),
            branch_scores=branch_scores,
            source_scores=source_scores,
            source_ranks=source_ranks,
            score=boost * (max(branch_scores.values()) + 0.15 * sum(branch_scores.values())),
            provenance=tuple(sorted(branch_scores)),
        ))
    return sorted(merged, key=lambda item: (-item.score, item.frame.video_id, item.frame.frame_idx, item.frame.frame_id))


def _candidate_score(candidate: RetrievalCandidate, rank: int) -> float:
    for value in (candidate.final_score, candidate.reranker_score, candidate.fusion_score):
        if value is not None:
            return float(value)
    return 1.0 / rank
