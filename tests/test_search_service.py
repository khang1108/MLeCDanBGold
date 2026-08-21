from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from hcmai.common.schemas import (
    FrameRecord,
    RetrievalCandidate,
    RetrievalSource,
    SearchRequest,
    TRAKERequest,
    TaskType,
)
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import (
    SearchService,
    SearchServiceUnavailableError,
)
from hcmai.retrieval.retriever.pipeline import RetrievalService


class Data:
    def __init__(self) -> None:
        self.frame = FrameRecord(
            frame_id="f1",
            video_id="official-video",
            frame_idx=42,
            timestamp_ms=1_000,
            image_path="f1.jpg",
            width=640,
            height=360,
        )

    def get_frame(self, frame_id: str):
        if frame_id != "f1":
            raise KeyError(frame_id)
        return self.frame

    def get_evidence(self, frame_id, source):
        del frame_id, source
        return None

class Retrieval:
    last_query_encoding_ms = 1
    last_index_search_ms = 2

    def __init__(self, frame_id: str = "f1") -> None:
        self.frame_id = frame_id

    def search(self, query, top_k, filters, query_type):
        return [
            RetrievalCandidate(
                frame_id=self.frame_id,
                source_scores={RetrievalSource.VISUAL: 0.5},
                final_score=0.5,
            )
        ]


def test_materialization_uses_only_canonical_data() -> None:
    response = SearchService(
        cast(DataService, Data()), cast(RetrievalService, Retrieval())
    ).search(
        SearchRequest(query="red bus")
    )
    result = response.results[0]
    assert response.search_id.startswith("search-")
    assert (result.video_id, result.frame_idx, result.timestamp_ms) == (
        "official-video",
        42,
        1_000,
    )


def test_unknown_frame_fails_closed() -> None:
    with pytest.raises(KeyError, match="missing"):
        SearchService(
            cast(DataService, Data()),
            cast(RetrievalService, Retrieval("missing")),
        ).search(
            SearchRequest(query="red bus")
        )


def test_trake_is_registered_before_dependency_readiness() -> None:
    service = SearchService(None, None)
    with pytest.raises(
        SearchServiceUnavailableError, match="Retriever not loaded"
    ):
        service.search(
            TRAKERequest(
                query="enter kitchen -> add butter",
                events=["enter kitchen", "add butter"],
            )
        )
