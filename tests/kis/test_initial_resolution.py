"""Tests for grounded initial resolution and safe fallback."""

from hcmai.kis.models import KISInitialResolution, KISInitialResolutionEvent
from hcmai.kis.resolution.initial import KISIntentResolver


class StubLLM:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error

    def generate_structured(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.output


def test_initial_resolution_preserves_original_query_and_provenance():
    query = "Người đàn ông bước vào phòng, sau đó lấy chiếc cốc"
    llm = StubLLM(
        KISInitialResolution(
            events=[
                KISInitialResolutionEvent(source_text="Người đàn ông bước vào phòng"),
                KISInitialResolutionEvent(source_text="sau đó lấy chiếc cốc"),
            ]
        )
    )
    intent = KISIntentResolver(llm).resolve_initial(query, revision=1)
    assert intent.query_text == query
    assert [e.text for e in intent.events] == [
        "Người đàn ông bước vào phòng",
        "sau đó lấy chiếc cốc",
    ]
    assert all(e.origin == "source" for e in intent.events)
    assert intent.events[0].source_provenance.start_char == 0


def test_invalid_grounding_falls_back_to_one_untouched_event():
    query = "Người đàn ông bước vào phòng"
    llm = StubLLM(
        KISInitialResolution(
            events=[KISInitialResolutionEvent(source_text="A man enters a room")]
        )
    )
    intent = KISIntentResolver(llm).resolve_initial(query, revision=1)
    assert [e.text for e in intent.events] == [query]
    assert intent.events[0].source_provenance.source_text == query
