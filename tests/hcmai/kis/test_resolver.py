"""Tests for KISIntentResolver and bounded KISInitialResolution canonicalization."""

from __future__ import annotations

from unittest.mock import Mock
import pytest

from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.inference.errors import InferenceUnavailableError
from hcmai.kis.models import (
    KISInitialResolution,
    KISInitialResolutionEvent,
)
from hcmai.kis.resolution.initial import KISIntentResolver, KISResolutionError


Q1 = (
    "A woman talks to a man in a kitchen. They move to the left. They take a white plate."
)


def test_REQ_001_kis_resolution_schema_omits_language() -> None:
    assert "language" not in KISInitialResolution.model_fields
    assert "language" not in KISInitialResolution.model_json_schema().get("properties", {})


def test_resolver_canonicalizes_initial_natural_resolution() -> None:
    """Verify resolver constructs canonical revision, IDs, empty entities, and temporal edges."""
    llm = Mock()
    llm.generate_structured.return_value = KISInitialResolution(
        events=[
            KISInitialResolutionEvent(source_text="A woman talks to a man in a kitchen."),
            KISInitialResolutionEvent(source_text="They move to the left."),
            KISInitialResolutionEvent(source_text="They take a white plate."),
        ],
    )

    resolver = KISIntentResolver(llm)
    intent = resolver.resolve_initial(Q1, revision=1)

    assert intent.revision == 1
    assert not hasattr(intent, "inputs")
    assert [event.id for event in intent.events] == ["E1", "E2", "E3"]
    assert intent.entities == []
    assert [(edge.source, edge.target) for edge in intent.temporal_edges] == [
        ("E1", "E2"),
        ("E2", "E3"),
    ]


def test_legacy_resolver_adapter_uses_explicit_revision_not_input_count() -> None:
    """The temporary clue-sequence adapter keeps revision server-owned."""
    llm = Mock()
    llm.generate_structured.return_value = KISInitialResolution(
        events=[KISInitialResolutionEvent(text="A woman enters.")],
    )

    intent = KISIntentResolver(llm).resolve(
        ["First clue.", "Second clue."], revision=9
    )

    assert intent.revision == 9
    assert not hasattr(intent, "inputs")
    assert intent.events[0].id == "E1"


def test_resolver_rejects_blank_initial_query_or_invalid_revision() -> None:
    """Blank initial natural text and non-positive revisions fail before LLM use."""
    llm = Mock()
    resolver = KISIntentResolver(llm)
    with pytest.raises(ValueError, match="non-empty"):
        resolver.resolve_initial("   ", revision=1)
    with pytest.raises(ValueError, match="at least 1"):
        resolver.resolve_initial("A woman talks.", revision=0)
    llm.generate_structured.assert_not_called()


def test_resolver_falls_back_when_provider_unavailable() -> None:
    """InferenceUnavailableError falls back safely to single unsegmented event."""
    llm = Mock()
    llm.generate_structured.side_effect = InferenceUnavailableError("GPU node down")
    resolver = KISIntentResolver(llm)
    intent = resolver.resolve_initial("A woman in a kitchen.", revision=1)
    assert len(intent.events) == 1
    assert intent.events[0].text == "A woman in a kitchen."
