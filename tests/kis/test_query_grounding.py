"""Tests for source grounding and language classification in KIS query hypotheses."""

from hcmai.kis.hypothesis.grounding import align_source_fragments, infer_query_language
from hcmai.kis.resolution.prompts import KIS_INITIAL_RESOLVER_SYSTEM_PROMPT


def test_align_source_fragments_is_left_to_right_for_repeated_text():
    query = "cốc rồi cốc rồi bàn"
    spans = align_source_fragments(query, ["cốc", "cốc", "bàn"])
    assert [(s.start_char, s.end_char) for s in spans] == [(0, 3), (8, 11), (16, 19)]


def test_align_source_fragments_rejects_paraphrase():
    try:
        align_source_fragments("người đàn ông vào phòng", ["a man enters a room"])
    except ValueError as exc:
        assert "not grounded" in str(exc)
    else:
        raise AssertionError("paraphrase must not be accepted as source grounding")


def test_infer_query_language_is_server_owned_and_bounded():
    assert infer_query_language("A person walks into a room") == "en"
    assert infer_query_language("Người đàn ông bước vào phòng") == "vi"
    assert infer_query_language("Người đàn ông picks up a cup") in {"vi", "mixed"}


# --- Task 4: Grounding prompt few-shot example must be verbatim ---

_EN_INPUT = (
    "The camera starts on a framed certificate, then pans right to "
    "a craftsperson engraving a metal plate with an unusual tool."
)


def _extract_english_example_events(prompt: str) -> list[str]:
    """Extract numbered event lines from the first (English) few-shot example."""
    lines = prompt.splitlines()
    in_example = False
    events: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Input:") and "framed certificate" in stripped:
            in_example = True
            events = []
            continue
        if in_example:
            if not stripped or stripped.startswith("Input:"):
                break
            if len(stripped) >= 3 and stripped[0].isdigit() and stripped[1] == ".":
                events.append(stripped[2:].strip())
    return events


def test_prompt_english_example_events_are_verbatim_substrings():
    """The English few-shot example output must be verbatim substrings of the input.

    This test prevents a regression where the prompt teaches the LLM to paraphrase
    instead of copy, which causes align_source_fragments() to raise ValueError on the
    LLM's own output.
    """
    events = _extract_english_example_events(KIS_INITIAL_RESOLVER_SYSTEM_PROMPT)
    assert len(events) >= 1, "Could not extract English example events from prompt"
    for event in events:
        assert event in _EN_INPUT, (
            f"Prompt example event is not a verbatim substring of the source.\n"
            f"Event:  {event!r}\n"
            f"Source: {_EN_INPUT!r}"
        )


def test_prompt_english_example_events_pass_align_source_fragments():
    """align_source_fragments must not raise for the prompt's English example."""
    events = _extract_english_example_events(KIS_INITIAL_RESOLVER_SYSTEM_PROMPT)
    assert len(events) >= 1
    spans = align_source_fragments(_EN_INPUT, events)
    assert len(spans) == len(events)
    for span in spans:
        assert span.source_text in _EN_INPUT


