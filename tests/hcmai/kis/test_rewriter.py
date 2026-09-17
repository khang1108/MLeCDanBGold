"""Tests for topology-preserving global KIS rewrites."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from hcmai.kis.models import (
    KISEntity,
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.kis.resolver import KISResolutionError
from hcmai.kis.rewriter import (
    GlobalRewriteEvent,
    GlobalRewriteResolution,
    KISGlobalRewriter,
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


def test_REQ_001_global_rewrite_does_not_expose_language() -> None:
    assert "language" not in GlobalRewriteResolution.model_fields
    assert "language" not in GlobalRewriteResolution.model_json_schema().get("properties", {})


def test_global_rewrite_preserves_ids_order_and_images() -> None:
    llm = Mock()
    llm.generate_structured.return_value = GlobalRewriteResolution(
        query_text="A woman talks to a man, then takes a plate.",
        entities=[],
        events=[
            GlobalRewriteEvent(event_id="E1", text="A woman talks to a man.", bindings=[]),
            GlobalRewriteEvent(event_id="E2", text="The woman takes a plate.", bindings=[]),
        ],
    )
    base = _base_multimodal_intent()
    rewritten = KISGlobalRewriter(llm).rewrite(base, "Resolve pronouns globally")

    assert rewritten.revision == base.revision + 1
    assert [event.id for event in rewritten.events] == ["E1", "E2"]
    assert [event.images for event in rewritten.events] == [
        base.events[0].images,
        base.events[1].images,
    ]
    assert [(edge.source, edge.target) for edge in rewritten.temporal_edges] == [("E1", "E2")]


def test_global_rewrite_rejects_different_event_count() -> None:
    llm = Mock()
    llm.generate_structured.return_value = GlobalRewriteResolution(
        query_text="Only one event",
        entities=[],
        events=[GlobalRewriteEvent(event_id="E1", text="one")],
    )

    with pytest.raises(KISResolutionError, match="count|exactly|event"):
        KISGlobalRewriter(llm).rewrite(_base_intent(), "Rewrite globally")


def test_global_rewrite_does_not_invent_text_for_image_only_event() -> None:
    base = _base_multimodal_intent().model_copy(
        update={
            "events": [
                _base_multimodal_intent().events[0].model_copy(update={"text": None}),
                _base_multimodal_intent().events[1],
            ],
            "query_text": "The woman takes a plate.",
        }
    )
    llm = Mock()
    llm.generate_structured.return_value = GlobalRewriteResolution(
        query_text="The woman takes a plate.",
        entities=[],
        events=[
            GlobalRewriteEvent(event_id="E1", text="Invented visual description"),
            GlobalRewriteEvent(event_id="E2", text="The woman takes a plate."),
        ],
    )

    with pytest.raises(KISResolutionError, match="image-only|text"):
        KISGlobalRewriter(llm).rewrite(base, "Rewrite globally")


def test_global_rewrite_preserves_revision_from_caller_not_model() -> None:
    llm = Mock()
    llm.generate_structured.return_value = GlobalRewriteResolution(
        query_text="A woman talks to a man.",
        entities=[],
        events=[GlobalRewriteEvent(event_id="E1", text="A woman talks to a man.")],
    )
    base = _base_intent().model_copy(update={"events": [ _base_intent().events[0] ], "temporal_edges": []})
    rewritten = KISGlobalRewriter(llm).rewrite(base, "rewrite")
    assert rewritten.revision == 2
