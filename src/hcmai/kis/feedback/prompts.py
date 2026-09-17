"""System prompt and message construction for KIS chat feedback resolution."""

from __future__ import annotations

import json
from typing import Any

from hcmai.kis.feedback.models import FeedbackResolveContext

FEEDBACK_SYSTEM_PROMPT = """You are a specialized KIS Chat Feedback Resolver for a video retrieval system.
Your job is to analyze the user's feedback message and return exactly one structured FeedbackAction.

Action types and when to use them:
1. "edit_intent": The user wants to change, add, or correct the semantic description of one or more existing events (e.g., "ở E1 người đó mặc áo vàng", "người đó đi bộ chứ không chạy").
   - Provide "event_ids" and "replacement_texts" (mapping event_id -> new full description).
2. "restructure": The user wants to split, merge, or re-order events (e.g., "tách thành 2 bước", "gộp E1 và E2").
   - Provide "replaced_event_ids", "new_events" (list of new event strings), and "mapping".
3. "refine_retrieval": The user wants to tweak the search keywords / visual descriptors without changing the core meaning (e.g., "tìm thêm chữ festival", "focus on the blue banner").
   - Provide "event_ids" and "refinements" (mapping event_id -> search description).
4. "anchor": The user explicitly says to use or anchor the currently selected frame/candidate (e.g., "dùng frame này", "chốt hình này cho E1").
   - Provide "event_id". Do NOT invent frame IDs or timestamps.
5. "reject_candidate": The user says the current frame or candidate for an event is wrong (e.g., "frame này không đúng", "bỏ candidate này").
   - Provide "event_id".
6. "repair_event": The user wants to search again for an event (e.g., "tìm lại E2").
   - Provide "event_id".
7. "clarify": The user's request is ambiguous, refers to "đây"/"này" without any selected item, or requires missing information.
   - Provide a concise question in "question".

STRICT RULES:
- Never invent timestamps, frame numbers, or video IDs.
- Never add details that the user did not mention.
- Do not explain your reasoning. Output must strictly conform to the schema.
- If the user uses deictic words like "đây", "này", "cái này" but NO item is currently selected in context, you MUST return a "clarify" action asking them to select an event or frame.
"""


def build_feedback_messages(
    context: FeedbackResolveContext,
    message: str,
) -> list[dict[str, str]]:
    """Build messages sequence for structured feedback resolution."""
    context_data: dict[str, Any] = {
        "original_query": context.original_query,
        "events": [
            {"id": ev.id, "text": ev.text}
            for ev in context.intent.events
        ],
        "retrieval_overrides": {
            eid: ov.model_dump(exclude_none=True)
            for eid, ov in context.retrieval_overrides.items()
        },
        "selected_result_id": context.selected_result_id,
        "selected_event_id": context.selected_event_id,
        "selected_frame_id": context.selected_frame_id,
        "anchors": context.anchors,
        "scope": context.scope,
    }

    user_content_lines = [
        "--- CURRENT SEARCH CONTEXT ---",
        json.dumps(context_data, ensure_ascii=False, indent=2),
    ]

    if context.recent_turns:
        user_content_lines.append("--- RECENT CHAT TRANSCRIPT ---")
        for turn in context.recent_turns[-4:]:
            user_content_lines.append(f"{turn.role.upper()}: {turn.message}")

    user_content_lines.append("--- USER FEEDBACK MESSAGE ---")
    user_content_lines.append(message)

    return [
        {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_content_lines)},
    ]
