"""Tests for Vietnamese BM25 and selected dense-query routing."""

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
            ("mot.",),
            [],
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
            ("mot.",),
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


def test_kis_bm25_does_not_require_query_preparation() -> None:
    """Search Vietnamese text directly when candidate generation is unavailable."""

    temporal = RecordingTemporal()
    pipeline = KISPipeline(
        corpus=cast(Any, object()),
        temporal=cast(Any, temporal),
        query_preparation=None,
    )

    response = pipeline.execute(
        SearchRequest(query="người đi xe máy", use_dense=False, use_bm25=True)
    )

    _, kwargs = temporal.calls[0]
    assert kwargs["caption_events"] == ("người đi xe máy",)
    assert response.bm25_caption_events == ["người đi xe máy"]


def test_trake_candidate_only_changes_dense_query() -> None:
    """Keep original Vietnamese events on BM25 when Dense uses a candidate."""

    temporal = RecordingTemporal()
    preparation = RecordingPreparation()
    pipeline = TRAKEPipeline(
        temporal=cast(Any, temporal),
        query_preparation=cast(Any, preparation),
    )

    response = pipeline.execute(
        TRAKERequest(
            events=["một người chạy"],
            retrieval_events=["a person running"],
            use_dense=True,
            use_bm25=True,
        )
    )

    original, kwargs = temporal.calls[0]
    assert original == ["một người chạy"]
    assert kwargs["retrieval_events"] == ["a person running"]
    assert kwargs["caption_events"] == ["một người chạy"]
    assert preparation.calls == []
    assert response.dense_events == ["a person running"]
    assert response.bm25_caption_events == ["một người chạy"]


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
