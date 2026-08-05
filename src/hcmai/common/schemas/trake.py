"""TRAKE multi-event query parsing contracts."""

from __future__ import annotations

from pydantic import Field

from .base import ContractModel, NonEmptyString


class TrakeParseInferenceRequest(ContractModel):
    """Complete bounded context required for one TRAKE parsing call."""

    instruction: NonEmptyString
    raw_query: NonEmptyString = Field(max_length=1_000)


class TrakeParseResponse(ContractModel):
    """Ordered atomic events split from one TRAKE query."""

    events: list[NonEmptyString] = Field(min_length=1, max_length=20)
