"""System prompt and message construction for KIS chat feedback resolution."""

from __future__ import annotations

import json
from typing import Any

from hcmai.kis.feedback.models import FeedbackResolveContext

FEEDBACK_SYSTEM_PROMPT = """You are a specialized KIS Chat Feedback Resolver for a video retrieval system.
Your job is to analyze the user's feedback message and return exactly one structured FeedbackAction.
CRITICAL: Every action object MUST include the "type" field with the exact action type string.

Action types and when to use them:
1. "refine_retrieval": The user wants to adjust, refine, or focus the search keywords or visual details for specific event(s) without changing the overall temporal storyline structure.
   - Provide "type": "refine_retrieval", "event_ids": list of event IDs, and "refinements": {event_id: "search description in English"}.
   - Examples:
     * "tập trung vào hình ảnh 2 đứa trẻ cầm banner" -> refine_retrieval for the relevant event (e.g. E1 or E2) with visual text: "two children holding a banner".
     * "tìm thêm chữ festival", "focus on the blue banner", "chú ý xe màu đỏ", "người mặc áo dài".
   - CRITICAL: Whenever the user provides new visual details, objects, people, colors, or search keywords (e.g. "tập trung vào...", "tìm theo...", "focus on...", "chú ý..."), you MUST use "refine_retrieval" or "edit_intent". NEVER use "repair_event" when descriptive keywords are provided!

2. "edit_intent": The user wants to change, add, or correct the core semantic definition of one or more existing events.
   - Provide "type": "edit_intent", "event_ids": list of event IDs, and "replacement_texts": {event_id: "new full description in English"}.
   - Examples: "ở E1 người đó mặc áo vàng", "người đó đi bộ chứ không chạy".

3. "restructure": The user wants to split, merge, or re-order events (e.g., "tách thành 2 bước", "gộp E1 và E2").
   - Provide "type": "restructure", "replaced_event_ids", "new_events", and "mapping".

4. "anchor": The user explicitly says to use or anchor the currently selected frame/candidate (e.g., "dùng frame này", "chốt hình này cho E1").
   - Provide "type": "anchor", "event_id". Do NOT invent frame IDs or timestamps.

5. "reject_candidate": The user says the current frame or candidate for an event is wrong (e.g., "frame này không đúng", "bỏ candidate này").
   - Provide "type": "reject_candidate", "event_id".

6. "repair_event": The user wants to search again for an event WITHOUT providing any new description (e.g., "tìm lại E2", "thử lại E1", "search E1 again").
   - Provide "type": "repair_event", "event_id".
   - CRITICAL: DO NOT use "repair_event" if the user provided search keywords or visual descriptions! Use "refine_retrieval"!

7. "clarify": The user's request is ambiguous, refers to "đây"/"này" without any selected item, or requires missing information.
   - Provide "type": "clarify", "question".

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
