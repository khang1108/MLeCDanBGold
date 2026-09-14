"""Prompt construction for literal one-to-one event translation."""

from collections.abc import Sequence

_TRANSLATE_SYSTEM_PROMPT = """You translate video-retrieval events into concise literal English.

Rules:
- Translate each event independently preserving exact event count and event order.
- Output event i must correspond directly to input event i.
- Do not merge, split, drop, duplicate, reorder, infer, explain, enrich, or add information.
- Preserve all people, objects, actions, colors, numbers, positions, and exact uppercase tokens or codes.
- Output valid JSON matching the schema with \"events\": list[str].
"""


def translation_messages(events: Sequence[str]) -> list[dict[str, str]]:
    """Build the structured LLM messages for one ordered event sequence."""
    user_content = "\n".join(
        f"Event {index}: {event}" for index, event in enumerate(events, start=1)
    )
    return [
        {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Translate the following events:\n{user_content}"},
    ]
