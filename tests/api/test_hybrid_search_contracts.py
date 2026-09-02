"""Contract tests for selectable hybrid temporal search modes."""

import pytest
from hcmai.api.contracts import SearchRequest, SearchResponse, TRAKERequest, TRAKEResponse
from hcmai.api.contracts.latency import SearchLatency
from pydantic import ValidationError


def _zero_latency() -> SearchLatency:
    return SearchLatency(
        query_ms=0,
        retrieval_ms=0,
        alignment_ms=0,
        materialization_ms=0,
        total_ms=0,
    )


def test_search_rejects_both_sources_off() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        SearchRequest(query="mot su kien", use_dense=False, use_bm25=False)


def test_trake_rejects_candidate_event_count_mismatch() -> None:
    with pytest.raises(ValidationError, match="retrieval_events"):
        TRAKERequest(events=["E1", "E2"], retrieval_events=["C1"])


def test_hybrid_request_defaults_enable_both_sources() -> None:
    request = SearchRequest(query="mot su kien")

    assert request.use_dense is True
    assert request.use_bm25 is True
    assert request.retrieval_events is None


def test_api_contracts_reject_more_than_32_selected_events() -> None:
    """Bound public temporal arrays before they reach orchestration."""

    events = [f"event {index}" for index in range(33)]
    with pytest.raises(ValidationError):
        SearchRequest(query="query", retrieval_events=events)
    with pytest.raises(ValidationError):
        TRAKERequest(events=events)


def test_bm25_only_response_omits_dense_events() -> None:
    response = SearchResponse(
        query="mot su kien",
        events=["mot su kien"],
        dense_events=None,
        bm25_caption_events=["an event"],
        use_dense=False,
        use_bm25=True,
        latency=_zero_latency(),
    )

    assert response.dense_events is None


@pytest.mark.parametrize("response_type", [SearchResponse, TRAKEResponse])
def test_bm25_response_requires_aligned_caption_events(response_type: type) -> None:
    payload = {
        "events": ["E1", "E2"],
        "dense_events": None,
        "use_dense": False,
        "use_bm25": True,
        "latency": _zero_latency(),
    }
    if response_type is SearchResponse:
        payload["query"] = "mot su kien"

    with pytest.raises(ValidationError, match="bm25_caption_events"):
        response_type(**payload)

    payload["bm25_caption_events"] = ["C1"]
    with pytest.raises(ValidationError, match="original event count"):
        response_type(**payload)
