"""Structured response schema for literal event translation only."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LiteralTranslation(BaseModel):
    """Ordered literal-English translations returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    events: list[NonBlank]
