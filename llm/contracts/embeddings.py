"""Bound hosted text-embedding requests and preserve ordered response vectors."""

from __future__ import annotations
from typing import Literal
from pydantic import Field
from .common import HTTPContract, NonEmptyString

class TextEmbeddingRequest(HTTPContract):
    """Accept up to 512 texts; encoder configuration controls inference batches."""

    model: NonEmptyString
    input: list[NonEmptyString] = Field(min_length=1, max_length=512)

class TextEmbeddingData(HTTPContract):
    object: Literal["embedding"] = "embedding"
    index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=1)

class TextEmbeddingResponse(HTTPContract):
    object: Literal["list"] = "list"
    model: NonEmptyString
    data: list[TextEmbeddingData]
