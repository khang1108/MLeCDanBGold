"""Behavioral tests for one-to-one event translation."""

from unittest.mock import Mock

import pytest

from hcmai.common.config import EventTranslationConfig
from hcmai.inference.config import ModelEndpointConfig
from hcmai.inference.errors import InferenceAuthError, InferenceUnavailableError
from hcmai.inference.clients.llm import LLMClient
from hcmai.retrieval.translation import EventTranslationError, EventTranslator


def _translator(llm: Mock, cache: object | None = None) -> EventTranslator:
    """Construct the translator with the minimum configuration it consumes."""
    config = Mock(
        prompt_version="event-translation-v1",
        cache_enabled=True,
        cache_ttl_seconds=3600,
        cache_max_entries=2048,
    )
    return EventTranslator(llm, config, cache=cache)


def test_english_events_bypass_llm() -> None:
    """English input remains normalized and does not invoke structured inference."""
    llm = Mock(model="test-model")

    result = _translator(llm).translate(("A woman enters",), language="en")

    assert result == ("A woman enters",)
    llm.generate_structured.assert_not_called()


def test_vietnamese_translation_preserves_event_count_and_order() -> None:
    """Non-English events map positionally to exactly one translated event each."""
    llm = Mock(model="test-model")
    llm.generate_structured.return_value = Mock(
        events=["A woman enters", "She closes the door"]
    )

    result = _translator(llm).translate(
        ("Một phụ nữ đi vào", "Cô ấy đóng cửa"), language="vi"
    )

    assert result == ("A woman enters", "She closes the door")


def test_translator_accepts_language_as_a_positional_argument() -> None:
    """The public translation contract accepts the brief's positional form."""
    llm = Mock(model="test-model")
    llm.generate_structured.return_value = Mock(events=["A woman enters"])

    result = _translator(llm).translate(("Một phụ nữ đi vào",), "vi")

    assert result == ("A woman enters",)


def test_translator_has_no_candidate_generation_api() -> None:
    """Literal translation cannot reintroduce retired query expansion."""
    assert getattr(EventTranslator, "generate_query_candidates", None) is None


def test_translation_preserves_required_uppercase_tokens() -> None:
    """Identifiers in the source event are not lost in translation."""
    llm = Mock(model="test-model")
    llm.generate_structured.return_value = Mock(events=["A woman holds OBJ_42"])

    result = _translator(llm).translate(("Người phụ nữ cầm OBJ_42",), language="vi")

    assert result == ("A woman holds OBJ_42",)


def test_translation_rejects_missing_required_uppercase_token() -> None:
    """A source identifier cannot be silently omitted from its translation."""
    llm = Mock(model="test-model")
    llm.generate_structured.return_value = Mock(events=["A woman holds a cup"])

    with pytest.raises(EventTranslationError, match="omitted required token 'OBJ_42'"):
        _translator(llm).translate(("Người phụ nữ cầm OBJ_42",), language="vi")


def test_translation_rejects_token_moved_to_a_different_event() -> None:
    """An identifier in another output event cannot satisfy positional fidelity."""
    llm = Mock(model="test-model")
    llm.generate_structured.return_value = Mock(
        events=["A woman holds OBJ_43", "She puts down OBJ_42"]
    )

    with pytest.raises(EventTranslationError, match="omitted required token 'OBJ_42'"):
        _translator(llm).translate(
            ("Người phụ nữ cầm OBJ_42", "Cô ấy đặt nó xuống"), language="vi"
        )


def test_cache_identity_changes_when_llm_model_changes() -> None:
    """Translations from differently named LLMs cannot share a cache entry."""
    first_llm = Mock(model="model-one")
    first_llm.generate_structured.return_value = Mock(events=["A woman enters"])
    second_llm = Mock(model="model-two")
    second_llm.generate_structured.return_value = Mock(events=["A woman enters"])

    class SharedCache:
        """Small in-memory cache exposing the translator cache protocol."""

        def __init__(self) -> None:
            """Initialize a hand-checkable cache fixture."""
            self.entries: dict[object, object] = {}

        def get(self, key: object) -> object | None:
            """Return the value previously stored under the exact key."""
            return self.entries.get(key)

        def put(self, key: object, value: object) -> None:
            """Store one value under the exact cache key."""
            self.entries[key] = value

    cache = SharedCache()

    _translator(first_llm, cache).translate(("Một phụ nữ đi vào",), language="vi")
    _translator(second_llm, cache).translate(("Một phụ nữ đi vào",), language="vi")

    assert first_llm.generate_structured.call_count == 1
    assert second_llm.generate_structured.call_count == 1


def test_cache_reuses_translation_for_the_same_llm_model() -> None:
    """Repeated input uses the entry keyed by the configured LLM model."""
    llm = Mock(model="provider/model-a")
    llm.generate_structured.return_value = Mock(events=["A woman enters"])
    translator = _translator(llm)

    first = translator.translate(("Một phụ nữ đi vào",), language="vi")
    second = translator.translate(("Một phụ nữ đi vào",), language="vi")

    assert first == second == ("A woman enters",)
    assert llm.generate_structured.call_count == 1


def test_cache_identity_isolates_prompt_versions_and_event_case() -> None:
    """Prompt revisions and case-distinct normalized events cannot share a cache entry."""
    llm = Mock(model="provider/model-a")
    llm.generate_structured.return_value = Mock(events=["A woman enters"])
    from hcmai.retrieval.translation.cache import EventTranslationCache

    cache = EventTranslationCache(max_entries=8, ttl_seconds=3600)
    first = EventTranslator(
        llm,
        EventTranslationConfig(prompt_version="event-translation-v1"),
        cache=cache,
    )
    second = EventTranslator(
        llm,
        EventTranslationConfig(prompt_version="event-translation-v2"),
        cache=cache,
    )

    first.translate(("một phụ nữ đi vào",), "vi")
    second.translate(("một phụ nữ đi vào",), "vi")
    second.translate(("Một phụ nữ đi vào",), "vi")

    assert llm.generate_structured.call_count == 3


def test_invalid_translation_raises_event_translation_error() -> None:
    """A response that changes cardinality fails at the translation boundary."""
    llm = Mock(model="test-model")
    llm.generate_structured.return_value = Mock(events=["Only one event"])

    with pytest.raises(EventTranslationError, match="changed event count"):
        _translator(llm).translate(("Một phụ nữ", "Cô ấy đi vào"), language="vi")


def test_malformed_real_client_schema_raises_event_translation_error() -> None:
    """Malformed structured LLM output crosses the translation error boundary."""
    transport = Mock()
    transport.post_json.return_value = {
        "choices": [{"message": {"content": '{"unexpected": "schema"}'}}]
    }
    llm = LLMClient(
        ModelEndpointConfig(base_url="https://api.example/v1", model="test-model"),
        transport=transport,
    )
    translator = EventTranslator(llm, EventTranslationConfig(cache_enabled=False))

    with pytest.raises(EventTranslationError, match="structured translation response"):
        translator.translate(("Một phụ nữ đi vào",), language="vi")


@pytest.mark.parametrize(
    "error",
    [InferenceAuthError("denied"), InferenceUnavailableError("offline")],
)
def test_provider_auth_and_unavailability_errors_are_not_wrapped(error: Exception) -> None:
    """Provider authentication and reachability retain their inference error class."""
    transport = Mock()
    transport.post_json.side_effect = error
    llm = LLMClient(
        ModelEndpointConfig(base_url="https://api.example/v1", model="test-model"),
        transport=transport,
    )
    translator = EventTranslator(llm, EventTranslationConfig(cache_enabled=False))

    with pytest.raises(type(error)):
        translator.translate(("Một phụ nữ đi vào",), language="vi")
