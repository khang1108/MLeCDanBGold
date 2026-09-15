"""Semantic graph models for KIS intents.

This module defines the domain models for structured KIS intent representation:
entities, events, entity bindings, temporal edges, and semantic resolution contracts.
It enforces graph validity, continuity, and canonical timeline order without depending
on HTTP transport schemas.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

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


class KISResolutionEntity(BaseModel):
    """Semantic entity returned by LLM without server-assigned ID."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["person", "object", "place", "text", "other"]
    description: NonBlank


class KISResolutionEvent(BaseModel):
    """Semantic event returned by LLM with zero-based entity index references."""

    model_config = ConfigDict(extra="forbid")
    text: NonBlank
    entity_indices: list[int] = Field(default_factory=list)


class KISResolution(BaseModel):
    """Semantic-only response requested from LLM without server-owned fields."""

    model_config = ConfigDict(extra="forbid")
    language: Literal["vi", "en"]
    query_text: NonBlank
    entities: list[KISResolutionEntity] = Field(default_factory=list)
    events: list[KISResolutionEvent] = Field(min_length=1)


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
    images: list[KISImageRef] = Field(default_factory=list)
    bindings: list[KISEntityBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Require every event to retain at least one source of semantic evidence."""
        if self.text is None and not self.images:
            raise ValueError("KIS event requires text or image evidence")
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
    language: Literal["vi", "en"] | None
    query_text: NonBlank | None
    entities: list[KISEntity] = Field(default_factory=list)
    events: list[KISEvent] = Field(min_length=1)
    temporal_edges: list[KISTemporalEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        """Validate identity uniqueness, referential integrity, and edge ordering."""
        if self.language is None or self.query_text is None:
            if self.language is not None or self.query_text is not None:
                raise ValueError("language and query_text must both be present or absent")
            if any(event.text is not None for event in self.events):
                raise ValueError(
                    "language and query_text may be absent only for image-only intents"
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
    "KISEntity",
    "KISEntityBinding",
    "KISEvent",
    "KISImageRef",
    "KISIntent",
    "KISResolution",
    "KISResolutionEntity",
    "KISResolutionEvent",
    "KISTemporalEdge",
]
