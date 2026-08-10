"""Training-free window localization using retrieval and lexical evidence."""

from __future__ import annotations

import re
from typing import Protocol

from .models import EvidenceBundle, LocalizedWindow, ParsedVQAQuery


class WindowLocalizer(Protocol):
    def localize(
        self, parsed: ParsedVQAQuery, bundles: list[EvidenceBundle], *, limit: int
    ) -> list[LocalizedWindow]: ...


class SimilarityLocalizer:
    """Cheap deterministic baseline with temporal and video diversity."""

    def __init__(self, *, overlap_ratio: float = 0.6) -> None:
        if not 0 <= overlap_ratio <= 1:
            raise ValueError("overlap_ratio must be between zero and one")
        self.overlap_ratio = overlap_ratio

    def localize(
        self, parsed: ParsedVQAQuery, bundles: list[EvidenceBundle], *, limit: int
    ) -> list[LocalizedWindow]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_terms = _terms(f"{parsed.retrieval_query} {parsed.question}")
        scored: list[LocalizedWindow] = []
        for bundle in bundles:
            evidence_terms = _terms(" ".join(item.value for item in bundle.items))
            lexical = len(query_terms & evidence_terms) / max(1, len(query_terms))
            score = bundle.window.score + 0.25 * lexical
            labels = ("retrieval_similarity",) + (("lexical_overlap",) if lexical else ())
            scored.append(LocalizedWindow(bundle=bundle, score=score, reason_labels=labels))
        selected: list[LocalizedWindow] = []
        for item in sorted(scored, key=lambda value: (-value.score, value.bundle.window.video_id, value.bundle.window.start_ms)):
            if any(_overlap(item, prior) > self.overlap_ratio for prior in selected):
                continue
            selected.append(item)
            if len(selected) == limit:
                break
        return selected


def _terms(value: str) -> set[str]:
    return {term for term in re.findall(r"\w+", value.casefold()) if len(term) > 1}


def _overlap(left: LocalizedWindow, right: LocalizedWindow) -> float:
    a, b = left.bundle.window, right.bundle.window
    if a.video_id != b.video_id:
        return 0.0
    intersection = max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))
    return intersection / max(1, min(a.end_ms - a.start_ms, b.end_ms - b.start_ms))
