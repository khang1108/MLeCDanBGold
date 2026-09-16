"""Structured interaction and latency logging for EventTrail."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from hcmai.common.utils.logging import get_logger

logger = get_logger("hcmai.event_trail.interactions")


def log_trail_event(
    *,
    event_type: str,
    kis_revision: int,
    snapshot_id: str | None = None,
    result_id: str | None = None,
    video_id: str | None = None,
    trail_session_id: str | None = None,
    trail_revision: int | None = None,
    event_id: str | None = None,
    search_session_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Emit one compact, structured JSON interaction record to the interactions logger."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "search_session_id": search_session_id,
        "kis_revision": kis_revision,
        "type": event_type,
        "snapshot_id": snapshot_id,
        "result_id": result_id,
        "video_id": video_id,
        "trail_session_id": trail_session_id,
        "trail_revision": trail_revision,
        "event_id": event_id,
        "payload": payload or {},
    }
    logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
