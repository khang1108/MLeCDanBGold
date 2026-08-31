"""Pydantic primitives shared only by offline artifact boundaries."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ContractModel(BaseModel):
    """Reject unknown artifact fields and normalize surrounding whitespace."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


__all__ = ["ContractModel", "NonEmptyString"]
