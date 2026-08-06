"""Variant-level retrieval fusion for the KIS golden path."""

from __future__ import annotations

from collections.abc import Sequence

from hcmai.common.schemas import RetrievalCandidate, RetrievalResult, RetrievalTrace
from hcmai.kis.variants import QueryVariant


def retrieve_query_variants(
    retrieval,
    variants: Sequence[QueryVariant],
    *,
    top_k: int,
    filters,
    query_type,
    rrf_k: int = 60,
) -> RetrievalResult:
    """Batch retrieval variants and fuse exact identities deterministically."""

    queries = [variant.query for variant in variants]
    raw_results = retrieval.search_batch(queries, top_k, filters, query_type)
    results = [
        result if isinstance(result, RetrievalResult)
        else RetrievalResult(candidates=result)
        for result in raw_results
    ]
    if len(results) != len(variants):
        raise ValueError("retrieval returned a different number of variant results")
    pool: dict[str, RetrievalCandidate] = {}
    scores: dict[str, float] = {}
    provenance: dict[str, list[dict[str, object]]] = {}
    first_seen: dict[str, int] = {}
    original_top = results[0].candidates[0].frame_id if results[0].candidates else None

    for variant_index, (variant, result) in enumerate(zip(variants, results)):
        for rank, candidate in enumerate(result.candidates, start=1):
            frame_id = candidate.frame_id
            first_seen.setdefault(frame_id, len(first_seen))
            pool.setdefault(frame_id, candidate.model_copy(deep=True))
            contribution = variant.weight / (rrf_k + rank)
            scores[frame_id] = scores.get(frame_id, 0.0) + contribution
            provenance.setdefault(frame_id, []).append({
                "variant_index": variant_index,
                "kind": variant.kind,
                "rank": rank,
                "weight": variant.weight,
            })

    ordered_ids = sorted(
        pool,
        key=lambda frame_id: (-scores[frame_id], first_seen[frame_id], frame_id),
    )
    if original_top is not None and original_top in ordered_ids:
        ordered_ids.remove(original_top)
        ordered_ids.insert(0, original_top)
    candidates = []
    for frame_id in ordered_ids[:top_k]:
        candidate = pool[frame_id]
        metadata = dict(candidate.metadata)
        metadata["query_variant_provenance"] = provenance[frame_id]
        candidates.append(candidate.model_copy(update={
            "fusion_score": scores[frame_id],
            "final_score": scores[frame_id],
            "metadata": metadata,
        }))

    trace = RetrievalTrace()
    warnings: list[str] = []
    first_candidate_ms = None
    for index, result in enumerate(results):
        trace = trace.merged(result.trace, prefix=f"variant_{index}")
        warnings.extend(result.warnings)
        if result.time_to_first_candidate_ms is not None:
            first_candidate_ms = (
                result.time_to_first_candidate_ms
                if first_candidate_ms is None
                else min(first_candidate_ms, result.time_to_first_candidate_ms)
            )
    return RetrievalResult(
        candidates=candidates,
        trace=trace,
        warnings=list(dict.fromkeys(warnings)),
        time_to_first_candidate_ms=first_candidate_ms,
    )
