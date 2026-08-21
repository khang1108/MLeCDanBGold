from __future__ import annotations

import json
import logging

from hcmai.common.schemas import StageStatus, StageTrace
from hcmai.common.observability import MetricsRegistry, REDACTED, safe_content
from hcmai.common.observability.tracing import log_stage


def _trace(status: StageStatus = StageStatus.SUCCESS) -> StageTrace:
    return StageTrace(
        stage="search",
        started_at=1,
        ended_at=1.01,
        duration_ms=10,
        status=status,
        error_category=("timeout" if status is StageStatus.PARTIAL else None),
        input_count=3,
        output_count=2,
        backend="visual",
        fallback_used=status is StageStatus.PARTIAL,
    )


def test_metrics_count_latency_and_bounded_failure_category() -> None:
    metrics = MetricsRegistry()
    metrics.observe_stage(_trace(StageStatus.PARTIAL))

    snapshot = metrics.snapshot()

    assert snapshot["stage_counts"] == {"search": 1}
    assert snapshot["failure_counts"] == {"search:timeout": 1}
    assert snapshot["latency_histograms_ms"] == {"search": {"10": 1}}


def test_structured_log_contains_counts_without_user_content(caplog) -> None:
    logger = logging.getLogger("observability-test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_stage(
            logger,
            request_id="request-1",
            task_type="kis",
            trace=_trace(),
        )

    payload = json.loads(caplog.records[-1].message)
    assert payload["input_count"] == 3
    assert payload["output_count"] == 2
    assert "query" not in payload
    assert "answer" not in payload
    assert "image" not in payload


def test_sensitive_content_is_redacted_unless_debug_is_explicit() -> None:
    assert safe_content("secret prompt") == REDACTED
    assert safe_content("  visible\n debug  ", debug=True) == "visible debug"
