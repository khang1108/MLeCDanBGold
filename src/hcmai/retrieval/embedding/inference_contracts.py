"""HTTP response contracts for hosted embedding inference.

These Pydantic models validate the external remote-embedding boundary. Internal
retrieval candidates remain frozen dataclasses in ``hcmai.retrieval.models``.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


_NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class EmbeddingResponse(BaseModel):
    """Validated matrix and provenance returned by an embedding endpoint."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model: _NonEmptyString
    revision: str | None = None
    dimension: int = Field(gt=0)
    normalized: bool
    item_ids: list[_NonEmptyString] | None = None
    embeddings: list[list[float]]
    latency_ms: float = Field(ge=0)

    @field_validator("embeddings")
    @classmethod
    def validate_shape(cls, values: list[list[float]]) -> list[list[float]]:
        """Require a non-empty rectangular embedding matrix."""

        if not values or not values[0]:
            raise ValueError("embeddings must be a non-empty matrix")
        if any(len(row) != len(values[0]) for row in values):
            raise ValueError("embedding rows must have equal dimensions")
        return values

    @model_validator(mode="after")
    def validate_metadata(self) -> Self:
        """Keep dimension and optional ordered item identity aligned."""

        if len(self.embeddings[0]) != self.dimension:
            raise ValueError("embedding dimension metadata does not match vectors")
        if self.item_ids is not None:
            if len(self.item_ids) != len(self.embeddings):
                raise ValueError("item/embedding count mismatch")
            if len(set(self.item_ids)) != len(self.item_ids):
                raise ValueError("embedding item_ids must be unique")
        return self


TextEmbeddingResponse = EmbeddingResponse


__all__ = ["EmbeddingResponse", "TextEmbeddingResponse"]
