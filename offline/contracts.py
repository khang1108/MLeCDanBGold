"""Pydantic primitives shared by offline artifact producers and readers.

These contracts deliberately stay outside ``hcmai.corpus`` so offline stages
can validate published artifacts without importing runtime-private modules.
"""

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
