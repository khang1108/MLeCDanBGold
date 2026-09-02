"""Tests for lazy translation and selected query routing."""

from __future__ import annotations

from typing import Any, cast

import pytest
from hcmai.api.contracts import SearchRequest, TRAKERequest
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.orchestration.workflows.trake import TRAKEPipeline


class RecordingTemporal:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def search(self, events: object, **kwargs: object) -> Any:
        self.calls.append((events, kwargs))
        return type("Result", (), {"paths": (), "retrieval_ms": 0.0, "alignment_ms": 0.0})()


class RecordingPreparation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def translate_literal(self, events: tuple[str, ...]) -> tuple[str, ...]:
        self.calls.append(events)
        return tuple(f"EN {event}" for event in events)


@pytest.mark.parametrize(
    ("search_request", "dense_events", "caption_events", "translation_calls"),
    [
        (SearchRequest(query="mot.", use_dense=True, use_bm25=False), ("mot.",), None, []),
        (
            SearchRequest(query="mot.", use_dense=False, use_bm25=True),
            ("mot.",),
            ("EN mot.",),
            [("mot.",)],
        ),
        (
            SearchRequest(
                query="mot.", retrieval_events=["candidate"], use_dense=True, use_bm25=False
            ),
            ("candidate",),
            None,
            [],
        ),
        (
            SearchRequest(
                query="mot.", retrieval_events=["candidate"], use_dense=False, use_bm25=True
            ),
            ("candidate",),
            ("candidate",),
            [],
),],)
def test_kis_lazy_translation_routing(
    search_request: SearchRequest,
    dense_events: tuple[str, ...],
    caption_events: tuple[str, ...] | None,
    translation_calls: list[tuple[str, ...]],
) -> None:
    temporal = RecordingTemporal()
    preparation = RecordingPreparation()
    pipeline = KISPipeline(
        corpus=cast(Any, object()),
        temporal=cast(Any, temporal),
        query_preparation=cast(Any, preparation),
    )

    response = pipeline.execute(search_request)

    original, kwargs = temporal.calls[0]
    assert original == ("mot.",)
    assert kwargs["retrieval_events"] == dense_events
    assert kwargs["caption_events"] == caption_events
    assert preparation.calls == translation_calls
    assert response.dense_events == (list(dense_events) if search_request.use_dense else None)
    assert response.bm25_caption_events == (list(caption_events) if caption_events else None)


def test_kis_configured_event_limit_runs_before_translation_and_search() -> None:
    """Reject an oversized split KIS query before remote preparation."""

    temporal = RecordingTemporal()
    preparation = RecordingPreparation()
    pipeline = KISPipeline(
        corpus=cast(Any, object()),
        temporal=cast(Any, temporal),
        query_preparation=cast(Any, preparation),
        max_temporal_event_count=2,
    )

    with pytest.raises(ValueError, match="at most 2 temporal events"):
        pipeline.execute(SearchRequest(query="one\ntwo\nthree"))

    assert preparation.calls == []
    assert temporal.calls == []


def test_trake_configured_event_limit_runs_before_translation_and_search() -> None:
    """Apply deployment limits to explicit TRAKE event arrays."""

    temporal = RecordingTemporal()
    preparation = RecordingPreparation()
    pipeline = TRAKEPipeline(
        temporal=cast(Any, temporal),
        query_preparation=cast(Any, preparation),
        max_temporal_event_count=2,
    )

    with pytest.raises(ValueError, match="at most 2 temporal events"):
        pipeline.execute(TRAKERequest(events=["one", "two", "three"]))

    assert preparation.calls == []
    assert temporal.calls == []
