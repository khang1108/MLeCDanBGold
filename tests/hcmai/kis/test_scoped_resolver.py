"""Tests for event-scoped KIS semantic resolution."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from hcmai.kis.models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.kis.parser import EventPatchInstruction
from hcmai.kis.resolution import (
    KISResolutionError,
    KISScopedResolver,
    ScopedResolutionBatch,
    ScopedResolvedEvent,
    apply_scoped_resolutions,
)


def _base_intent() -> KISIntent:
    return KISIntent(
        revision=1,
        query_text="A woman talks to a man, then takes a plate.",
        entities=[
            KISEntity(id="X1", kind="person", description="woman"),
            KISEntity(id="X2", kind="person", description="man"),
        ],
        events=[
            KISEvent(id="E1", text="A woman talks to a man.", bindings=[]),
            KISEvent(id="E2", text="The woman takes a plate.", bindings=[]),
        ],
        temporal_edges=[KISTemporalEdge(source="E1", target="E2", relation="before")],
    )


def _base_multimodal_intent() -> KISIntent:
    base = _base_intent()
    return base.model_copy(
        update={
            "events": [
                base.events[0].model_copy(
                    update={
                        "images": [
                            KISImageRef(asset_id="sha256:a", content_type="image/png")
                        ]
                    }
                ),
                base.events[1],
            ]
        }
    )


def test_REQ_001_scoped_batch_does_not_expose_language() -> None:
    assert "language" not in ScopedResolutionBatch.model_fields
    assert "language" not in ScopedResolutionBatch.model_json_schema().get("properties", {})


def test_REQ_003_scoped_response_without_language_updates_textual_intent() -> None:
    llm = Mock()
    llm.generate_structured.return_value = ScopedResolutionBatch(
        events=[ScopedResolvedEvent(event_id="E2", text="Con chó được cập nhật")]
    )
    result = KISScopedResolver(llm).resolve(
        _base_intent(),
        [EventPatchInstruction(event_id="E2", instruction="cập nhật con chó")],
    )
    updated = apply_scoped_resolutions(_base_intent(), result, revision=2)
    assert updated.events[1].text == "Con chó được cập nhật"
    assert "language" not in result.model_dump()


def test_scoped_resolver_returns_only_named_events() -> None:
    llm = Mock()
    llm.generate_structured.return_value = ScopedResolutionBatch(
        events=[
            ScopedResolvedEvent(
                event_id="E2",
                text="The woman talks to a chef.",
                bindings=[KISEntityBinding(entity_id="X1", role="speaker")],
            )
        ],
    )
    resolver = KISScopedResolver(llm)

    result = resolver.resolve(
        _base_intent(),
        [EventPatchInstruction(event_id="E2", instruction="chef")],
    )

    assert [event.event_id for event in result.events] == ["E2"]
    assert result.events[0].text == "The woman talks to a chef."
    assert result.events[0].bindings == [KISEntityBinding(entity_id="X1", role="speaker")]


def test_scoped_resolver_rejects_model_widening_scope() -> None:
    llm = Mock()
    llm.generate_structured.return_value = ScopedResolutionBatch(
        events=[
            ScopedResolvedEvent(event_id="E1", text="unrelated"),
            ScopedResolvedEvent(event_id="E2", text="updated"),
        ],
    )

    with pytest.raises(KISResolutionError, match="exactly|scope|target"):
        KISScopedResolver(llm).resolve(
            _base_intent(),
            [EventPatchInstruction(event_id="E2", instruction="chef")],
        )


def test_apply_scoped_preserves_unrelated_event_and_images() -> None:
    base = _base_multimodal_intent()
    resolved = ScopedResolutionBatch(
        events=[ScopedResolvedEvent(event_id="E2", text="Updated plate action")],
    )

    result = apply_scoped_resolutions(base, resolved, revision=2)

    assert result.revision == 2
    assert result.events[0] == base.events[0]
    assert result.events[0].images == base.events[0].images
    assert result.events[1].text == "Updated plate action"
    assert result.events[1].images == base.events[1].images
    assert [(edge.source, edge.target) for edge in result.temporal_edges] == [("E1", "E2")]
    assert result.query_text == "A woman talks to a man. Updated plate action"


def test_first_textual_scoped_update_populates_query_text_for_image_only_intent() -> None:
    base = KISIntent(
        revision=1,
        query_text=None,
        entities=[],
        events=[KISEvent(id="E1", text=None, images=[KISImageRef(asset_id="a", content_type="image/png")])],
        temporal_edges=[],
    )
    resolved = ScopedResolutionBatch(
        events=[ScopedResolvedEvent(event_id="E1", text="The woman is holding a plate")],
    )

    result = apply_scoped_resolutions(base, resolved, revision=2)

    assert result.query_text == "The woman is holding a plate"
    assert result.events[0].text == "The woman is holding a plate"
    assert result.events[0].images == base.events[0].images


def test_scoped_resolver_rejects_unknown_binding_entity_and_blank_role() -> None:
    llm = Mock()
    llm.generate_structured.return_value = ScopedResolutionBatch(
        events=[
            ScopedResolvedEvent(
                event_id="E1",
                text="updated",
                bindings=[KISEntityBinding(entity_id="X9", role="speaker")],
            )
        ],
    )
    with pytest.raises(KISResolutionError, match="unknown entity"):
        KISScopedResolver(llm).resolve(
            _base_intent(), [EventPatchInstruction(event_id="E1", instruction="update")]
        )


def test_initial_scoped_batch_has_contiguous_events_and_no_bindings() -> None:
    llm = Mock()
    llm.generate_structured.return_value = ScopedResolutionBatch(
        events=[
            ScopedResolvedEvent(event_id="E1", text="Mot nguoi di bo"),
            ScopedResolvedEvent(event_id="E2", text="Nguoi ay ngoi xuong"),
        ],
    )

    result = KISScopedResolver(llm).resolve(
        None,
        [
            EventPatchInstruction(event_id="E1", instruction="di bo"),
            EventPatchInstruction(event_id="E2", instruction="ngoi"),
        ],
    )
    intent = apply_scoped_resolutions(None, result, revision=1)

    assert [event.id for event in intent.events] == ["E1", "E2"]
    assert all(not event.bindings for event in intent.events)
    assert [(edge.source, edge.target) for edge in intent.temporal_edges] == [("E1", "E2")]
    assert intent.revision == 1


def test_appended_text_event_preserves_validated_binding() -> None:
    base = _base_intent()
    resolved = ScopedResolutionBatch.model_validate({
        "events": [{
            "event_id": "E3",
            "text": "The same woman lifts a plate.",
            "bindings": [{"entity_id": "X1", "role": "actor"}],
        }],
    })
    updated = apply_scoped_resolutions(base, resolved, revision=base.revision + 1)
    assert len(updated.events) == 3
    assert updated.events[2].id == "E3"
    assert len(updated.events[2].bindings) == 1
    assert updated.events[2].bindings[0].entity_id == "X1"
    assert updated.events[2].bindings[0].role == "actor"
