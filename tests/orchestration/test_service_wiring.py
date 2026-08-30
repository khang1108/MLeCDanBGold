"""Composition tests for explicit KIS and TRAKE service wiring."""

from __future__ import annotations

from typing import Any, cast

from hcmai.orchestration.pipeline import SearchService


class _Data:
    """Placeholder canonical-data dependency used only for composition."""

    record_count = 0

    def has_evidence(self, source: object) -> bool:
        """Report no optional evidence for health composition."""

        del source
        return False


class _Retrieval:
    """Placeholder retrieval dependency used only for composition."""


def test_kis_and_trake_share_one_temporal_service() -> None:
    """Build both explicit workflows over exactly one temporal facade."""

    service = SearchService(
        data=cast(Any, _Data()),
        retrieval=cast(Any, _Retrieval()),
    )

    assert service.kis.temporal is service.trake.temporal
    assert not hasattr(service, "pipeline_registry")


def test_health_uses_loaded_dependencies_not_task_registration() -> None:
    """Report explicit task readiness without a generic query-type registry."""

    ready = SearchService(
        data=cast(Any, _Data()),
        retrieval=cast(Any, _Retrieval()),
    ).health()["capabilities"]
    unavailable = SearchService(data=None, retrieval=None).health()["capabilities"]

    assert (ready["search"], ready["kis"], ready["trake"]) == (True, True, True)
    assert (unavailable["search"], unavailable["kis"], unavailable["trake"]) == (
        False,
        False,
        False,
    )
    assert "query_types" not in ready
