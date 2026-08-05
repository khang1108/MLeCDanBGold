"""Provider contracts for bounded query suggestions."""

from __future__ import annotations

from typing import Protocol

from hcmai.common.schemas import (
    QuerySuggestionInferenceRequest,
    QuerySuggestionResponse,
)


class SuggestionAdapter(Protocol):
    def suggest(
        self, request: QuerySuggestionInferenceRequest
    ) -> QuerySuggestionResponse: ...

    def close(self) -> None: ...
