"""Fallback window localization when the shared temporal core is unavailable."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Protocol

from ..domain.models import EvidenceBundle, ParsedVQAQuery


class WindowLocalizer(Protocol):
    def localize(
        self, parsed: ParsedVQAQuery, bundles: list[EvidenceBundle], *, limit: int
    ) -> list[EvidenceBundle]: ...


class SimilarityLocalizer:
    """Cheap deterministic baseline with temporal and video diversity."""

    def __init__(self, *, overlap_ratio: float = 0.6) -> None:
        if not 0 <= overlap_ratio <= 1:
            raise ValueError("overlap_ratio must be between zero and one")
        self.overlap_ratio = overlap_ratio

    def localize(
        self, parsed: ParsedVQAQuery, bundles: list[EvidenceBundle], *, limit: int
    ) -> list[EvidenceBundle]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_terms = _terms(f"{parsed.retrieval_query} {parsed.question}")
        scored: list[EvidenceBundle] = []
        for bundle in bundles:
            evidence_terms = _terms(" ".join(item.value for item in bundle.items))
            lexical = len(query_terms & evidence_terms) / max(1, len(query_terms))
            score = bundle.scene.final_score + 0.25 * lexical
            labels = ("retrieval_similarity",) + (("lexical_overlap",) if lexical else ())
            scored.append(replace(
                bundle,
                scene=bundle.scene.model_copy(update={
                    "semantic_score": lexical,
                    "final_score": score,
                    "reason_labels": labels,
                }),
            ))
        selected: list[EvidenceBundle] = []
        for item in sorted(
            scored,
            key=lambda value: (
                -value.scene.final_score,
                value.scene.video_id,
                value.scene.start_ms,
            ),
        ):
            if any(_overlap(item, prior) > self.overlap_ratio for prior in selected):
                continue
            selected.append(item)
            if len(selected) == limit:
                break
        return selected


def _terms(value: str) -> set[str]:
    return {term for term in re.findall(r"\w+", value.casefold()) if len(term) > 1}


def _overlap(left: EvidenceBundle, right: EvidenceBundle) -> float:
    a, b = left.scene, right.scene
    if a.video_id != b.video_id:
        return 0.0
    intersection = max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))
    return intersection / max(1, min(a.end_ms - a.start_ms, b.end_ms - b.start_ms))
