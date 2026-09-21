"""Bounded LLM feedback resolver mapping user chat messages into typed actions."""

from __future__ import annotations

import re
from typing import Any

from hcmai.inference.clients.llm import LLMClient
from hcmai.kis.feedback.models import (
    ClarifyAction,
    FeedbackAction,
    FeedbackResolution,
    FeedbackResolveContext,
)
from hcmai.kis.feedback.prompts import build_feedback_messages


class FeedbackResolverError(Exception):
    """Exception raised when feedback resolution or validation fails."""


def validate_action_references(
    action: FeedbackAction,
    context: FeedbackResolveContext,
) -> None:
    """Validate that action only targets existing events and does not invent external identity."""
    valid_event_ids = {event.id for event in context.intent.events}

    if action.type == "query_edit_proposal":
        act_dict = action.action if isinstance(action.action, dict) else {}
        for key in ("event_id", "left_event_id", "right_event_id"):
            eid = act_dict.get(key)
            if eid and eid not in valid_event_ids:
                raise FeedbackResolverError(
                    f"Unknown event ID '{eid}' in query_edit_proposal action"
                )
        if "event_ids" in act_dict and isinstance(act_dict["event_ids"], (list, tuple)):
            for eid in act_dict["event_ids"]:
                if eid not in valid_event_ids:
                    raise FeedbackResolverError(
                        f"Unknown event ID '{eid}' in query_edit_proposal action"
                    )

    elif action.type == "refine_retrieval":
        for eid in action.event_ids:
            if eid not in valid_event_ids:
                raise FeedbackResolverError(
                    f"Unknown event ID '{eid}' in refine_retrieval action"
                )

    elif getattr(action, "event_id", None) is not None:
        if action.event_id not in valid_event_ids:
            raise FeedbackResolverError(
                f"Unknown event ID '{action.event_id}' in {action.type} action"
            )


class FeedbackResolver:
    """Resolver converting natural user chat feedback into single typed FeedbackAction."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def resolve(
        self,
        context: FeedbackResolveContext,
        message: str,
    ) -> FeedbackAction:
        """Resolve message within bounded context and return validated action."""
        stripped = message.strip()
        lower = stripped.lower()

        # Check for ungrounded deictic references when no selection is present in context
        has_selection = bool(
            context.selected_event_id
            or context.selected_frame_id
            or context.selected_result_id
        )
        deictic_patterns = [
            r"\bđây\b",
            r"\bở đây\b",
            r"\bcái này\b",
            r"\bframe này\b",
            r"\bhình này\b",
            r"\bchỗ này\b",
            r"\bđoạn này\b",
        ]
        if not has_selection and any(re.search(pat, lower) for pat in deictic_patterns):
            return ClarifyAction(
                question="Vui lòng chọn event hoặc frame cụ thể để áp dụng phản hồi này."
            )

        messages = build_feedback_messages(context, stripped)
        resolution = self._llm.generate_structured(
            messages,
            FeedbackResolution,
            temperature=0.0,
            max_tokens=1024,
        )
        action = resolution.action
        validate_action_references(action, context)
        return action


__all__ = [
    "FeedbackResolver",
    "FeedbackResolverError",
    "validate_action_references",
]
