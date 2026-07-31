"""Prompting and strict parsing for bounded query suggestions."""

from __future__ import annotations

import json
from typing import Any

from hcmai.common.schemas import QuerySuggestion

INSTRUCTION = """
Generate exactly the requested number of diverse, standalone visual-search
queries from the original query. Preserve every stated name, number, color,
negation, and event order. Never invent an object, action, setting, text, or
spoken evidence. Use Vietnamese and English where faithful. Cover useful
focuses from literal, action, subject, object, scene, and temporal.
Return only JSON: {"suggestions":[{"query":"...","language":"vi|en|mixed",
"focus":"literal|action|subject|object|scene|temporal"}]}. No Markdown.
""".strip()


def suggestion_messages(query: str, count: int) -> list[dict[str, Any]]:
    payload = json.dumps(
        {"original_query": query, "count": count}, ensure_ascii=False
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": payload}]},
    ]


def parse_suggestions(text: str, original: str, count: int) -> list[QuerySuggestion]:
    raw = _json_object(text).get("suggestions")
    if not isinstance(raw, list):
        raise ValueError("query-suggestion model did not return a suggestions list")
    output: list[QuerySuggestion] = []
    seen = {" ".join(original.lower().split())}
    for value in raw:
        if not isinstance(value, dict):
            continue
        query = str(value.get("query", "")).strip()
        normalized = " ".join(query.lower().split())
        if not query or normalized in seen:
            continue
        seen.add(normalized)
        output.append(QuerySuggestion(
            suggestion_id=f"suggestion-{len(output) + 1}",
            query=query,
            language=value.get("language", ""),
            focus=value.get("focus", ""),
        ))
        if len(output) == count:
            break
    if len(output) != count:
        raise ValueError(
            f"query-suggestion model returned {len(output)} unique items; "
            f"expected {count}"
        )
    return output


def _json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("query-suggestion model did not return a JSON object")
