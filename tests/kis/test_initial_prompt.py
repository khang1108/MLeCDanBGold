from hcmai.kis.prompts import (
    KIS_INITIAL_RESOLVER_SYSTEM_PROMPT,
    build_kis_initial_messages,
)


def test_initial_prompt_defines_event_as_retrievable_visual_moment_not_shot() -> None:
    prompt = KIS_INITIAL_RESOLVER_SYSTEM_PROMPT.lower()
    assert "retrievable visual moment" in prompt
    assert "continuous camera" in prompt
    assert "separate events" in prompt


def test_initial_prompt_does_not_request_entity_graph_work() -> None:
    prompt = KIS_INITIAL_RESOLVER_SYSTEM_PROMPT.lower()
    forbidden = (
        "entity tracking",
        "entity_indices",
        "query_text",
        "contradictions & corrections",
    )
    assert all(term not in prompt for term in forbidden)


def test_initial_messages_keep_one_narrative_as_input_without_pre_splitting() -> None:
    messages = build_kis_initial_messages((
        "The camera pans from a framed photograph to an artisan drawing a portrait.",
    ))
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Clue 1:" in messages[1]["content"]
