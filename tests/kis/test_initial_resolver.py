from hcmai.kis.models import KISInitialResolution, KISInitialResolutionEvent
from hcmai.kis.resolver import KISIntentResolver


class CapturingLLM:
    def __init__(self, result: KISInitialResolution) -> None:
        self.result = result
        self.calls = []

    def generate_structured(
        self,
        messages,
        response_model,
        *,
        temperature=0.0,
        max_tokens=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "response_model": response_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.result


def test_initial_resolver_uses_event_only_schema_and_512_token_budget() -> None:
    llm = CapturingLLM(
        KISInitialResolution(
            events=[KISInitialResolutionEvent(text="A framed photograph is visible.")]
        )
    )
    KISIntentResolver(llm).resolve_initial("Một bức ảnh trên bàn.", revision=1)

    call = llm.calls[0]
    assert call["response_model"] is KISInitialResolution
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 512


def test_initial_resolver_canonicalizes_two_events_without_entities() -> None:
    llm = CapturingLLM(
        KISInitialResolution(
            events=[
                KISInitialResolutionEvent(
                    text="A photograph shows a person shaking hands with Ho Chi Minh."
                ),
                KISInitialResolutionEvent(
                    text="An artisan draws that person's portrait with an unusual pen."
                ),
            ]
        )
    )

    intent = KISIntentResolver(llm).resolve_initial("ignored by fake", revision=4)

    assert intent.revision == 4
    assert intent.entities == []
    assert [event.id for event in intent.events] == ["E1", "E2"]
    assert [event.bindings for event in intent.events] == [[], []]
    assert intent.query_text == (
        "A photograph shows a person shaking hands with Ho Chi Minh. "
        "An artisan draws that person's portrait with an unusual pen."
    )
    assert [(edge.source, edge.target) for edge in intent.temporal_edges] == [
        ("E1", "E2")
    ]
