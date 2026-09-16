"""Tests for apply_scoped_resolutions and scoped resolver binding semantics."""

import pytest

from hcmai.kis.models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.kis.scoped_resolver import (
    ScopedResolutionBatch,
    apply_scoped_resolutions,
)


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


def test_appended_text_event_preserves_validated_binding(base_intent: KISIntent) -> None:
    resolved = ScopedResolutionBatch.model_validate({
        "language": "en",
        "events": [{
            "event_id": "E2",
            "text": "The same woman lifts a plate.",
            "bindings": [{"entity_id": "X1", "role": "actor"}],
        }],
    })
    updated = apply_scoped_resolutions(base_intent, resolved, revision=base_intent.revision + 1)
    assert len(updated.events) == 2
    assert updated.events[1].id == "E2"
    assert len(updated.events[1].bindings) == 1
    assert updated.events[1].bindings[0].entity_id == "X1"
    assert updated.events[1].bindings[0].role == "actor"
