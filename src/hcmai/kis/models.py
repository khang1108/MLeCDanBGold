"""Semantic graph models for KIS intents.

This module defines the domain models for structured KIS intent representation:
entities, events, entity bindings, and temporal edges. It enforces graph validity,
continuity, and canonical timeline order without depending on HTTP transport schemas.
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


class KISEvent(BaseModel):
    """A distinct timeline moment or action node."""

    model_config = ConfigDict(extra="forbid")

    id: EventId
    text: NonBlank
    bindings: list[KISEntityBinding] = Field(default_factory=list)


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
    inputs: list[NonBlank] = Field(min_length=1)
    language: Literal["vi", "en"]
    query_text: NonBlank
    entities: list[KISEntity] = Field(default_factory=list)
    events: list[KISEvent] = Field(min_length=1)
    temporal_edges: list[KISTemporalEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        """Validate identity uniqueness, referential integrity, and edge ordering."""
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

        # 5. Temporal edge validation
        for edge in self.temporal_edges:
            if edge.source not in event_indices:
                raise ValueError(
                    f"Temporal edge references unknown source event: {edge.source}"
                )
            if edge.target not in event_indices:
                raise ValueError(
                    f"Temporal edge references unknown target event: {edge.target}"
                )
            if edge.source == edge.target:
                raise ValueError(
                    f"Self temporal edge is invalid: {edge.source} -> {edge.target}"
                )
            if event_indices[edge.source] >= event_indices[edge.target]:
                raise ValueError(
                    f"Temporal edge {edge.source} before {edge.target} contradicts "
                    f"canonical event order"
                )

        return self
