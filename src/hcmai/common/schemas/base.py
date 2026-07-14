from __future__ import annotations

from typing import Annotated
from pydantic import (
    StringConstraints, 
    ConfigDict,
    BaseModel
)
NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

class ContractModel(BaseModel):
    """Base model for all shared project contracts."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )