"""Shared helper for optional DRES result logging across search routes.

This module logs search candidates to DRES in best-effort mode without affecting
successful user query responses.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from fastapi import Response

from hcmai.common.utils.logging import get_logger
from hcmai.vbs.models import (
    ApiClientAnswer,
    QueryEvent,
    QueryResultLog,
    RankedAnswer,
)

logger = get_logger(__name__)


def _now_ms() -> int:
    """Return the wall-clock Unix epoch used by DRES event and result logs."""
    return int(time.time() * 1000)


async def record_dres_result_log(
    service_container: dict[str, Any],
    response: Response,
    *,
    user_id: str | None,
    category: str,
    event_value: str,
    results: list[Any],
    rank_offset: int = 0,
    now_fn: Callable[[], int] | None = None,
) -> None:
    """Send an optional result log without changing successful retrieval."""
    log_status = "skipped"
    try:
        vbs_service = service_container.get("vbs_service")
        if (
            user_id is not None
            and user_id.strip()
            and vbs_service is not None
            and vbs_service.session_status(user_id).get("connected")
        ):
            evaluation_id = await vbs_service.resolve_evaluation(user_id)
            timestamp = (now_fn or _now_ms)()
            ranked = [
                RankedAnswer(
                    rank=rank_offset + index + 1,
                    answer=ApiClientAnswer(
                        media_item_name=vbs_service.media_item_name(result.video_id),
                        start=result.timestamp_ms,
                        end=result.timestamp_ms,
                    ),
                )
                for index, result in enumerate(results)
            ]
            payload = QueryResultLog(
                timestamp=timestamp,
                sort_type="list",
                result_set_availability="",
                results=ranked,
                events=[QueryEvent(
                    timestamp=timestamp,
                    category=category,
                    event_type="SEARCH",
                    value=event_value,
                )],
            )
            await vbs_service.log_results(user_id, evaluation_id, payload)
            log_status = "sent"
    except Exception as error:
        # DRES is an optional secondary side effect; never mask retrieval.
        logger.warning(
            "DRES result logging failed status=failed error_type=%s",
            type(error).__name__,
        )
        log_status = "failed"
    response.headers["X-DRES-Log-Status"] = log_status
