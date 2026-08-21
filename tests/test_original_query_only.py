from __future__ import annotations

import importlib.util

import hcmai.common.schemas as schemas


def test_retired_query_expansion_modules_are_absent() -> None:
    assert importlib.util.find_spec("hcmai.pipelines.kis.variants") is None
    assert importlib.util.find_spec("hcmai.pipelines.kis.retrieval") is None
    assert importlib.util.find_spec("hcmai.common.schemas.query_suggestion") is None


def test_query_suggestion_contracts_are_not_public() -> None:
    retired_names = {
        "QuerySuggestion",
        "QuerySuggestionInferenceRequest",
        "QuerySuggestionRequest",
        "QuerySuggestionResponse",
    }
    assert retired_names.isdisjoint(schemas.__all__)
