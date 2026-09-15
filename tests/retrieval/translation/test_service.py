"""Contract tests for the retrieval-owned event translation service.

These tests keep the S0 public test path aligned with the implementation plan.
They cover literal translation only; query-candidate generation is retired.
"""

from unittest.mock import Mock

import pytest

from hcmai.common.config import EventTranslationConfig
from hcmai.retrieval.translation.models import LiteralTranslation
from hcmai.retrieval.translation.service import EventTranslationError, EventTranslator


def test_translator_returns_english_input_without_llm_call() -> None:
    """English events remain normalized without structured inference."""
    llm = Mock(model="model-a")
    service = EventTranslator(llm, EventTranslationConfig())

    assert service.translate(["woman enters", "she sits"], "en") == (
        "woman enters",
        "she sits",
    )
    llm.generate_structured.assert_not_called()


def test_translator_uses_actual_llm_model_in_cache_identity() -> None:
    """Repeated non-English input reuses the entry for the active model."""
    llm = Mock(model="provider/model-a")
    llm.generate_structured.return_value = LiteralTranslation(events=["a woman enters"])
    service = EventTranslator(llm, EventTranslationConfig())

    first = service.translate(["một phụ nữ bước vào"], "vi")
    second = service.translate(["một phụ nữ bước vào"], "vi")

    assert first == second == ("a woman enters",)
    assert llm.generate_structured.call_count == 1


def test_translator_rejects_event_count_mismatch() -> None:
    """Translation cannot merge or drop canonical retrieval events."""
    llm = Mock(model="model-a")
    llm.generate_structured.return_value = LiteralTranslation(events=["a woman enters"])
    service = EventTranslator(llm, EventTranslationConfig())

    with pytest.raises(EventTranslationError, match="changed event count"):
        service.translate(["một phụ nữ", "cô ấy bước vào"], "vi")


def test_translator_has_no_candidate_generation_api() -> None:
    """The retired query-expansion capability cannot re-enter this service."""
    assert not hasattr(EventTranslator, "generate_query_candidates")
