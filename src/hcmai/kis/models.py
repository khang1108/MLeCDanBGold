"""Semantic graph models for KIS intents.

This module defines the domain models for structured KIS intent representation:
entities, events, entity bindings, temporal edges, and semantic resolution contracts.
It enforces graph validity, continuity, and canonical timeline order without depending
on HTTP transport schemas.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EntityId = Annotated[str, StringConstraints(pattern=r"^X[1-9]\d*$")]
EventId = Annotated[str, StringConstraints(pattern=r"^E[1-9]\d*$")]
InitialEventText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
]


class SourceProvenance(BaseModel):
    """Source grounding span within the canonical original natural-language query."""

    model_config = ConfigDict(extra="forbid")

    source_text: NonBlank
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("source provenance end_char must exceed start_char")
        return self


class KISInitialResolutionEvent(BaseModel):
    """One concise retrievable visual moment returned by the initial resolver."""

    model_config = ConfigDict(extra="forbid")
    source_text: InitialEventText = Field(
        description=(
            "Verbatim source-language fragment for one retrievable chronological moment."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def accept_text_field(cls, data: Any) -> Any:
        if isinstance(data, dict) and "text" in data and "source_text" not in data:
            data = dict(data)
            data["source_text"] = data.pop("text")
        return data

    @property
    def text(self) -> str:
        return self.source_text


class KISInitialResolution(BaseModel):
    """Event-only output contract for initial natural-language KIS resolution."""

    model_config = ConfigDict(extra="forbid")
    events: list[KISInitialResolutionEvent] = Field(
        min_length=1,
        max_length=DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
        description="Chronologically ordered distinct retrievable visual moments.",
    )



class KISEntity(BaseModel):
    """An entity discovered or tracked across KIS clue revisions."""

    model_config = ConfigDict(extra="forbid")

    id: EntityId
    kind: Literal["person", "object", "place", "text", "other"]
    description: NonBlank


class KISEntityBinding(BaseModel):
    """Associates an entity with a specific semantic role in an event."""

    model_config = ConfigDict(extra="forbid")

    entity_id: EntityId
    role: NonBlank


class KISImageRef(BaseModel):
    """A validated reference to a user-provided image stored outside the intent."""

    model_config = ConfigDict(extra="forbid")

    asset_id: NonBlank
    content_type: Literal["image/jpeg", "image/png", "image/webp"]


class KISEvent(BaseModel):
    """A distinct timeline moment with text and/or attached image evidence."""

    model_config = ConfigDict(extra="forbid")

    id: EventId
    text: NonBlank | None = None
    source_provenance: SourceProvenance | None = None
    origin: Literal["source", "user_override", "user_added"] = "source"
    images: list[KISImageRef] = Field(default_factory=list)
    bindings: list[KISEntityBinding] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_text_origin(cls, data: Any) -> Any:
        """Mark ungrounded legacy text as overridden instead of source-grounded.

        Initial source-grounded resolution supplies both provenance and an explicit
        source origin. Older callers that provide only text cannot truthfully claim
        source grounding, so keep them valid while making that distinction explicit.
        """
        if (
            isinstance(data, dict)
            and "origin" not in data
            and data.get("text") is not None
            and data.get("source_provenance") is None
        ):
            data = dict(data)
            data["origin"] = "user_override"
        return data

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Require every event to retain at least one source of semantic evidence."""
        if self.text is None and not self.images:
            raise ValueError("KIS event requires text or image evidence")
        if self.origin == "user_added" and self.source_provenance is not None:
            raise ValueError("user-added events cannot retain source provenance")
        # Keep compatibility with older image/text fixtures that omitted an
        # explicit origin, while rejecting explicitly source-grounded text that
        # cannot be traced to a source span.
        if (
            self.origin == "source"
            and self.text is not None
            and self.source_provenance is None
        ):
            raise ValueError("source-grounded text requires source provenance")
        if (
            self.origin == "source"
            and self.text is not None
            and self.source_provenance is not None
            and self.text != self.source_provenance.source_text
        ):
            raise ValueError("source text must match source provenance")
        return self


class KISTemporalEdge(BaseModel):
    """A directed temporal relationship between two events."""

    model_config = ConfigDict(extra="forbid")

    source: EventId
    relation: Literal["before"] = "before"
    target: EventId


class KISIntent(BaseModel):
    """Semantic graph representing a resolved KIS multi-clue search intent."""

    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=1)
    query_text: NonBlank | None
    language: Literal["vi", "en", "mixed"] = "en"
    entities: list[KISEntity] = Field(default_factory=list)
    events: list[KISEvent] = Field(min_length=1)
    temporal_edges: list[KISTemporalEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        """Validate identity uniqueness, referential integrity, and edge ordering."""
        has_text = any(event.text is not None for event in self.events)
        if has_text != (self.query_text is not None):
            raise ValueError(
                "query_text must be present exactly when the intent contains text"
            )

        # 1. Unique entity IDs
        entity_ids = [entity.id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Duplicate entity IDs found in intent")
        known_entities = set(entity_ids)

        # 2. Maximum temporal event cardinality
        if len(self.events) > DEFAULT_MAX_TEMPORAL_EVENT_COUNT:
            raise ValueError(
                f"Event count ({len(self.events)}) exceeds maximum allowed "
                f"({DEFAULT_MAX_TEMPORAL_EVENT_COUNT})"
            )

        # 3. Sequential event IDs E1..En
        expected_ids = [f"E{i+1}" for i in range(len(self.events))]
        actual_ids = [event.id for event in self.events]
        if actual_ids != expected_ids:
            raise ValueError(
                f"Event IDs must be sequentially ordered E1..En, got: {actual_ids}"
            )
        event_indices = {event_id: idx for idx, event_id in enumerate(actual_ids)}

        # 4. Entity binding referential integrity
        for event in self.events:
            for binding in event.bindings:
                if binding.entity_id not in known_entities:
                    raise ValueError(
                        f"Event {event.id} references unknown entity: {binding.entity_id}"
                    )

        # 5. Temporal edge validation: require complete sequential adjacent chain
        expected_edges = [
            (f"E{i}", f"E{i+1}") for i in range(1, len(self.events))
        ]
        actual_edges = [(edge.source, edge.target) for edge in self.temporal_edges]
        if actual_edges != expected_edges:
            raise ValueError(
                f"Temporal edges must form complete sequential adjacent chain {expected_edges}, got: {actual_edges}"
            )

        return self


__all__ = [
    "InitialEventText",
    "KISEntity",
    "KISEntityBinding",
    "KISEvent",
    "KISImageRef",
    "KISInitialResolution",
    "KISInitialResolutionEvent",
    "KISIntent",
    "KISTemporalEdge",
    "SourceProvenance",
]
