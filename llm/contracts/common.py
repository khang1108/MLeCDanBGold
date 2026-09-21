from __future__ import annotations
from typing import Annotated
from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class HTTPContract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, populate_by_name=True)

__all__ = ["HTTPContract", "NonEmptyString"]
