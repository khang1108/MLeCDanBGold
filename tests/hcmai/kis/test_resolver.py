"""Tests for KISIntentResolver and semantic-only KISResolution canonicalization."""

from __future__ import annotations

from unittest.mock import Mock
import pytest

from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.inference.errors import InferenceUnavailableError
from hcmai.kis.models import (
    KISIntent,
    KISResolution,
    KISResolutionEntity,
    KISResolutionEvent,
)
from hcmai.kis.resolver import KISIntentResolver, KISResolutionError


Q1 = "A woman is standing in a kitchen."
Q2 = "She is talking to a man."
Q3 = "Before taking a white plate, they move to the left."


def test_REQ_001_kis_resolution_schema_omits_language() -> None:
    assert "language" not in KISResolution.model_fields
    assert "language" not in KISResolution.model_json_schema().get("properties", {})


def test_resolver_canonicalizes_initial_natural_resolution() -> None:
    """Verify resolver constructs canonical revision, IDs, bindings, and temporal edges."""
    llm = Mock()
    llm.generate_structured.return_value = KISResolution(
        query_text="A woman talks to a man, moves to the left, and takes a white plate.",
        entities=[
            KISResolutionEntity(kind="person", description="woman in kitchen"),
            KISResolutionEntity(kind="person", description="man"),
            KISResolutionEntity(kind="object", description="white plate"),
        ],
        events=[
            KISResolutionEvent(text="A woman talks to a man", entity_indices=[0, 1]),
            KISResolutionEvent(text="They move to the left", entity_indices=[0, 1]),
            KISResolutionEvent(text="They take a white plate", entity_indices=[0, 1, 2]),
        ],
    )

    resolver = KISIntentResolver(llm)
    intent = resolver.resolve_initial(Q1, revision=1)

    assert intent.revision == 1
    assert not hasattr(intent, "inputs")
    assert [event.id for event in intent.events] == ["E1", "E2", "E3"]
    assert [(e.source, e.target) for e in intent.temporal_edges] == [
        ("E1", "E2"),
        ("E2", "E3"),
    ]
    assert [entity.id for entity in intent.entities] == ["X1", "X2", "X3"]
    assert intent.events[0].bindings[0].entity_id == "X1"
    assert intent.events[0].bindings[1].entity_id == "X2"
    assert intent.events[2].bindings[2].entity_id == "X3"

    llm.generate_structured.assert_called_once()
    messages, response_model = llm.generate_structured.call_args.args
    assert response_model is KISResolution
    # Prompt should not ask model for revision, inputs, or IDs
    assert "revision" not in messages[0]["content"]


def test_legacy_resolver_adapter_uses_explicit_revision_not_input_count() -> None:
    """The temporary clue-sequence adapter keeps revision server-owned."""
    llm = Mock()
    llm.generate_structured.return_value = KISResolution(
        query_text="A woman enters.",
        entities=[],
        events=[KISResolutionEvent(text="A woman enters")],
    )

    intent = KISIntentResolver(llm).resolve(
        ["First clue.", "Second clue."], revision=9
    )

    assert intent.revision == 9
    assert not hasattr(intent, "inputs")


def test_resolver_rejects_out_of_range_entity_index() -> None:
    """Out-of-range entity index raises KISResolutionError."""
    llm = Mock()
    llm.generate_structured.return_value = KISResolution(
        query_text="A woman talks to someone.",
        entities=[KISResolutionEntity(kind="person", description="woman")],
        events=[KISResolutionEvent(text="talks", entity_indices=[99])],
    )
    resolver = KISIntentResolver(llm)
    with pytest.raises(KISResolutionError, match="out-of-range"):
        resolver.resolve_initial("A woman talks.", revision=1)


def test_resolver_rejects_duplicate_entity_index_in_event() -> None:
    """Duplicate entity index in single event raises KISResolutionError."""
    llm = Mock()
    llm.generate_structured.return_value = KISResolution(
        query_text="A woman talks to herself.",
        entities=[KISResolutionEntity(kind="person", description="woman")],
        events=[KISResolutionEvent(text="talks", entity_indices=[0, 0])],
    )
    resolver = KISIntentResolver(llm)
    with pytest.raises(KISResolutionError, match="duplicate"):
        resolver.resolve_initial("A woman talks.", revision=1)


def test_resolver_rejects_too_many_events() -> None:
    """More events than DEFAULT_MAX_TEMPORAL_EVENT_COUNT raises KISResolutionError."""
    llm = Mock()
    events = [
        KISResolutionEvent(text=f"event {i}", entity_indices=[])
        for i in range(DEFAULT_MAX_TEMPORAL_EVENT_COUNT + 1)
    ]
    llm.generate_structured.return_value = KISResolution(
        query_text="Too many events.",
        entities=[],
        events=events,
    )
    resolver = KISIntentResolver(llm)
    with pytest.raises(KISResolutionError):
        resolver.resolve_initial("Too many events.", revision=1)


def test_resolver_rejects_blank_initial_query_or_invalid_revision() -> None:
    """Blank initial natural text and non-positive revisions fail before LLM use."""
    llm = Mock()
    resolver = KISIntentResolver(llm)
    with pytest.raises(ValueError, match="non-empty"):
        resolver.resolve_initial("   ", revision=1)
    with pytest.raises(ValueError, match="at least 1"):
        resolver.resolve_initial("A woman talks.", revision=0)
    llm.generate_structured.assert_not_called()


def test_resolver_propagates_provider_unavailable() -> None:
    """InferenceUnavailableError propagates directly without domain wrapping."""
    llm = Mock()
    llm.generate_structured.side_effect = InferenceUnavailableError("GPU node down")
    resolver = KISIntentResolver(llm)
    with pytest.raises(InferenceUnavailableError):
        resolver.resolve_initial("A woman in a kitchen.", revision=1)
