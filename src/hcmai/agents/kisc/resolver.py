"""Provider-independent bounded conversation interpretation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from hcmai.common.schemas import (
    ConversationState,
    ConversationTurn,
    FrameFeedback,
)

StructuredCall = Callable[[dict[str, Any]], object]

_INSTRUCTION = """
Return exactly one complete ConversationState, never a delta. Include the full
standalone_query, positive_constraints, negative_constraints,
uncertain_constraints, accepted_frame_ids, and rejected_frame_ids. Preserve
compatible additions. Apply newest-wins corrections: replace stale corrected
constraints and do not retain contradictions that were superseded. Put explicit
negations in negative_constraints and unresolved ambiguity in
uncertain_constraints. Current feedback is the newest event: accepted frames
must leave rejected_frame_ids, and rejected frames must leave
accepted_frame_ids. Return no search result, SearchRequest, retrieval action,
tool action, plan, or chain-of-thought. Do not retrieve, use tools, use ReAct,
or make recursive calls. Write standalone_query as a concise English visual
search description. Translate the user's stated evidence faithfully, preserve
names and numbers, and never invent an object, action, setting, or event.
""".strip()


class ConversationResolverError(RuntimeError):
    """Bounded failure at the conversation interpretation boundary."""


def _bounded_error(prefix: str, error: Exception) -> str:
    detail = " ".join(str(error).split())[:160]
    return f"{prefix}: {detail or type(error).__name__}"


def _merged_feedback(
    previous: ConversationState | None,
    feedback: FrameFeedback | None,
) -> tuple[list[str], list[str]]:
    accepted = list(previous.accepted_frame_ids) if previous else []
    rejected = list(previous.rejected_frame_ids) if previous else []
    if feedback is None:
        return accepted, rejected
    for frame_id in feedback.accepted_frame_ids:
        rejected = [item for item in rejected if item != frame_id]
        if frame_id not in accepted:
            accepted.append(frame_id)
    for frame_id in feedback.rejected_frame_ids:
        accepted = [item for item in accepted if item != frame_id]
        if frame_id not in rejected:
            rejected.append(frame_id)
    return accepted, rejected


def _request(
    history: Sequence[ConversationTurn],
    current_message: str,
    feedback: FrameFeedback | None,
    previous: ConversationState | None,
) -> dict[str, Any]:
    return {
        "instruction": _INSTRUCTION,
        "history": [turn.model_dump(mode="json") for turn in history],
        "current_message": current_message,
        "feedback": feedback.model_dump(mode="json") if feedback else None,
        "previous_state": (previous.model_dump(mode="json") if previous else None),
        "response_schema": ConversationState.model_json_schema(),
    }


def _state_from_output(output: object) -> ConversationState:
    if isinstance(output, ConversationState):
        return output.model_copy(deep=True)
    if not isinstance(output, Mapping):
        raise ConversationResolverError(
            "structured output must be a ConversationState or mapping"
        )
    missing = set(ConversationState.model_fields) - set(output)
    if missing:
        raise ConversationResolverError(
            "structured output is incomplete: " + ", ".join(sorted(missing))
        )
    try:
        return ConversationState.model_validate(output)
    except ValidationError as error:
        raise ConversationResolverError(
            "structured output failed ConversationState validation"
        ) from error


class ConversationResolver:
    """Resolve explicit bounded context with one injected structured call."""

    def __init__(self, structured_call: StructuredCall) -> None:
        self.structured_call = structured_call

    def resolve(
        self,
        history: Sequence[ConversationTurn],
        current_message: str,
        feedback: FrameFeedback | None = None,
        previous_state: ConversationState | None = None,
    ) -> ConversationState:
        """Return one validated complete interpreted state."""
        message = current_message.strip()
        if not message:
            raise ConversationResolverError("current_message must not be empty")
        request = _request(history, message, feedback, previous_state)
        try:
            output = self.structured_call(request)
        except Exception as error:
            raise ConversationResolverError(
                _bounded_error("structured provider failed", error)
            ) from error
        state = _state_from_output(output)
        accepted, rejected = _merged_feedback(previous_state, feedback)
        if state.accepted_frame_ids != accepted or state.rejected_frame_ids != rejected:
            raise ConversationResolverError(
                "structured output violated newest-wins feedback state"
            )
        return state
