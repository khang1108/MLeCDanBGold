"""System prompt and message construction for KIS chat feedback resolution."""

from __future__ import annotations

import json
from typing import Any

from hcmai.kis.feedback.models import FeedbackResolveContext

FEEDBACK_SYSTEM_PROMPT = """You are a specialized KIS Chat Feedback Resolver for a video retrieval system.
Your job is to analyze the user's feedback message and return exactly one structured FeedbackAction.
CRITICAL: Every action object MUST include the "type" field with the exact action type string.

Action types and when to use them:
1. "query_edit_proposal": When the user asks to split, merge, reorder, add, or semantically edit canonical events.
   - Provide "type": "query_edit_proposal", "action": dictionary describing the query action (e.g. {"type": "split", "event_id": "E1", ...}, {"type": "edit", "event_id": "E1", "text": "..."}), and "explanation": short description of the proposal.
   - Never directly apply topology changes.

2. "refine_retrieval": The user wants to adjust, refine, or focus the search keywords or visual details for specific event(s) without changing the overall temporal storyline structure.
   - Provide "type": "refine_retrieval", "event_ids": list of event IDs, and "refinements": {event_id: "search description"}.
   - Examples: "tập trung vào hình ảnh 2 đứa trẻ cầm banner", "tìm thêm chữ festival", "focus on the blue banner", "chú ý xe màu đỏ".

3. "repair_event": The user wants to search again for an event WITHOUT providing any new description (e.g., "tìm lại E2", "thử lại E1", "search E1 again").
   - Provide "type": "repair_event", "event_id".

4. "clarify": The user's request is ambiguous, refers to "đây"/"này" without any selected item, or requires missing information.
   - Provide "type": "clarify", "question".

STRICT RULES:
- When the user asks to split, merge, reorder, add, or semantically edit canonical events, return query_edit_proposal. Never directly apply topology changes.
- Retrieval-only wording may use refine_retrieval.
- Result-path anchors/rejections are handled by Hypothesis Explorer, not generic chat.
- Never invent timestamps, frame numbers, or video IDs.
- Never add details that the user did not mention.
- Do not explain your reasoning outside the schema. Output must strictly conform to the schema.
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
