"""Structured interaction and latency logging for KIS Query Hypotheses.

This module logs query hypothesis lifecycle operations (preview, commit, undo)
in a structured JSON format for replayable experiments without leaking raw reasoning.
"""

from __future__ import annotations

import json
from typing import Sequence

from hcmai.common.utils.logging import get_logger

logger = get_logger("hcmai.kis.hypothesis.interactions")


def log_query_hypothesis_event(
    *,
    event_type: str,
    session_id: str,
    query_revision: int,
    action_type: str,
    affected_event_ids: Sequence[str],
    latency_ms: float,
    committed: bool = True,
    action_payload: dict[str, Any] | None = None,
) -> None:
    """Emit a structured JSON record for query hypothesis operations (preview, commit, undo)."""
    payload = {
        "event_type": event_type,
        "session_id": session_id,
        "query_revision": query_revision,
        "action_type": action_type,
        "affected_event_ids": list(affected_event_ids),
        "latency_ms": round(latency_ms, 3),
        "committed": committed,
        "action_payload": action_payload or {},
    }
    logger.info("query_hypothesis_event %s", json.dumps(payload, sort_keys=True))


def extract_action_metadata(action: Any) -> tuple[str, list[str], dict[str, Any]]:
    """Extract action type, affected event IDs, and minimal replay payload from a QueryHypothesisAction."""
    from hcmai.kis.hypothesis.models import (
        AddEvent,
        AttachImage,
        DetachImage,
        EditEvent,
        MergeEvents,
        ReorderEvents,
        SplitEvent,
    )

    if isinstance(action, SplitEvent):
        return (
            "split",
            [action.event_id],
            {
                "split_at": action.split_at,
                "image_assignments": {
                    k: list(v) for k, v in action.image_assignments.items()
                },
            },
        )
    if isinstance(action, MergeEvents):
        return (
            "merge",
            [action.left_event_id, action.right_event_id],
            {
                "left_event_id": action.left_event_id,
                "right_event_id": action.right_event_id,
            },
        )
    if isinstance(action, ReorderEvents):
        return (
            "reorder",
            list(action.event_ids),
            {"event_ids": list(action.event_ids)},
        )
    if isinstance(action, EditEvent):
        return (
            "edit",
            [action.event_id],
            {"text": action.text},
        )
    if isinstance(action, AddEvent):
        return (
            "add",
            [],
            {
                "position": action.position,
                "text": action.text,
                "image_count": len(action.images),
            },
        )
    if isinstance(action, AttachImage):
        return (
            "attach_image",
            [action.event_id],
            {"asset_id": action.image.asset_id},
        )
    if isinstance(action, DetachImage):
        return (
            "detach_image",
            [action.event_id],
            {"asset_id": action.asset_id},
        )
    return getattr(action, "type", "unknown"), [], {}
