"""Transition Diagnostic Dataset schema, loader, and counterfactual generator.

Verifies Phase 9 and Phase 10 of SOICT motion graph implementation plan:
Provides structured event trail annotations with state transitions, object manipulation,
direction changes, and counterfactual variants (reversal, adjacent swap).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hcmai.temporal.candidate_recall import EventAnnotation


@dataclass(frozen=True, slots=True)
class DiagnosticEvent:
    """One annotated semantic event within a diagnostic query."""

    id: str
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    target_frame_idxs: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
        }
        if self.start_ms is not None:
            data["start_ms"] = self.start_ms
        if self.end_ms is not None:
            data["end_ms"] = self.end_ms
        if self.target_frame_idxs:
            data["target_frame_idxs"] = list(self.target_frame_idxs)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticEvent:
        return cls(
            id=data["id"],
            text=data["text"],
            start_ms=data.get("start_ms"),
            end_ms=data.get("end_ms"),
            target_frame_idxs=tuple(data.get("target_frame_idxs", ())),
        )


@dataclass(frozen=True, slots=True)
class DiagnosticQuery:
    """One diagnostic multi-event query paired with its canonical ground truth video."""

    query_id: str
    video_id: str
    query: str
    category: str
    events: tuple[DiagnosticEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "video_id": self.video_id,
            "query": self.query,
            "category": self.category,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticQuery:
        events = tuple(DiagnosticEvent.from_dict(e) for e in data.get("events", []))
        return cls(
            query_id=data["query_id"],
            video_id=data["video_id"],
            query=data["query"],
            category=data.get("category", "general"),
            events=events,
        )

    def reversed_query(self) -> DiagnosticQuery:
        """Create a counterfactual query with chronological event order strictly reversed."""
        rev_events = tuple(reversed(self.events))
        rev_text = " then ".join(e.text for e in rev_events)
        return DiagnosticQuery(
            query_id=f"{self.query_id}_reverse",
            video_id=self.video_id,
            query=f"[REVERSE] {rev_text}",
            category=f"{self.category}_reverse",
            events=rev_events,
        )

    def swapped_query(self, swap_index: int = 0) -> DiagnosticQuery:
        """Create a counterfactual query with two adjacent events swapped."""
        if len(self.events) < 2:
            return self
        ev_list = list(self.events)
        idx = min(swap_index, len(ev_list) - 2)
        ev_list[idx], ev_list[idx + 1] = ev_list[idx + 1], ev_list[idx]
        swapped_text = " then ".join(e.text for e in ev_list)
        return DiagnosticQuery(
            query_id=f"{self.query_id}_swap_{idx}",
            video_id=self.video_id,
            query=f"[SWAP_{idx}] {swapped_text}",
            category=f"{self.category}_swap",
            events=tuple(ev_list),
        )


@dataclass(frozen=True, slots=True)
class DiagnosticDataset:
    """Collection of diagnostic queries for transition retrieval evaluation."""

    queries: tuple[DiagnosticQuery, ...]

    def __len__(self) -> int:
        return len(self.queries)

    def to_event_annotations(self) -> list[EventAnnotation]:
        """Convert all queries into candidate recall event annotations."""
        annotations: list[EventAnnotation] = []
        for q in self.queries:
            for idx, ev in enumerate(q.events):
                annotations.append(
                    EventAnnotation(
                        video_id=q.video_id,
                        event_index=idx,
                        target_frame_idxs=ev.target_frame_idxs,
                        start_ms=ev.start_ms,
                        end_ms=ev.end_ms,
                    )
                )
        return annotations

    @classmethod
    def load_json(cls, path: Path | str) -> DiagnosticDataset:
        """Load dataset from JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("queries", [])
        queries = tuple(DiagnosticQuery.from_dict(item) for item in items)
        return cls(queries=queries)

    def save_json(self, path: Path | str) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"queries": [q.to_dict() for q in self.queries]}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


__all__ = [
    "DiagnosticDataset",
    "DiagnosticEvent",
    "DiagnosticQuery",
]
