"""Tests for multimodal KIS prerequisites: canonicalization and atomic append."""

from unittest.mock import Mock

import pytest

from hcmai.api.contracts.kis import (
    EventPatch,
    InitialResolveOperation,
    KISSearchRequest,
    PatchEventsOperation,
)
from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
from hcmai.kis.models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.workflows.kis import KISSearchExecution


@pytest.fixture
def image_store() -> Mock:
    store = Mock()
    store.ref.side_effect = lambda asset_id: {
        "img_abc": KISImageRef(asset_id="img_abc", content_type="image/png"),
        "img_plate": KISImageRef(asset_id="img_plate", content_type="image/png"),
    }[asset_id]
    return store


@pytest.fixture
def base_intent() -> KISIntent:
    return KISIntent(
        revision=1,
        language="en",
        query_text="A woman stands in a kitchen.",
        entities=[
            KISEntity(id="X1", kind="person", description="woman in kitchen"),
        ],
        events=[
            KISEvent(
                id="E1",
                text="A woman stands in a kitchen.",
                bindings=[KISEntityBinding(entity_id="X1", role="actor")],
            ),
        ],
        temporal_edges=[],
    )


@pytest.fixture
def search_service(image_store: Mock) -> SearchService:
    service = SearchService(
        corpus=Mock(),
        retrieval=Mock(),
        temporal_evidence=Mock(),
        intent_resolver=Mock(),
        scoped_resolver=Mock(),
        global_rewriter=Mock(),
        event_translator=Mock(),
        kis_image_assets=image_store,
    )
    service.kis = Mock()
    service.kis.execute.return_value = KISSearchExecution(
        results=[
            SearchResult(
                frame_id="v1_f1",
                video_id="v1",
                frame_idx=10,
                timestamp_ms=1000,
                score=0.9,
                frame_ids=["v1_f1"],
                timestamps_ms=[1000],
                metadata=SearchResultMetadata(),
            )
        ],
        latency=SearchLatency(query_ms=1.0, retrieval_ms=1.0),
    )
    return service


def test_initial_resolve_image_canonicalization_overrides_client_metadata(
    search_service: SearchService, image_store: Mock
) -> None:
    request = KISSearchRequest(
        base_intent=None,
        expected_revision=0,
        operation=InitialResolveOperation(
            kind="initial_resolve",
            image_refs=[KISImageRef(asset_id="img_abc", content_type="image/jpeg")],
        ),
        use_dense=False,
        use_bm25=False,
        top_k=10,
    )
    response = search_service.search_kis(request)
    committed_image = response.intent.events[0].images[0]
    assert committed_image.asset_id == "img_abc"
    assert committed_image.content_type == "image/png"


def test_patch_can_append_image_only_event_atomically(
    search_service: SearchService, base_intent: KISIntent, image_store: Mock
) -> None:
    request = KISSearchRequest(
        base_intent=base_intent,
        expected_revision=1,
        operation=PatchEventsOperation(
            kind="patch_events",
            patches=[EventPatch(event_id="E2", add_image_ids=["img_plate"])],
        ),
        use_dense=True,
        use_bm25=True,
        top_k=10,
    )
    response = search_service.search_kis(request)
    assert len(response.intent.events) == 2
    event = response.intent.events[1]
    assert event.id == "E2"
    assert event.text is None
    assert [image.asset_id for image in event.images] == ["img_plate"]
    assert event.images[0].content_type == "image/png"
