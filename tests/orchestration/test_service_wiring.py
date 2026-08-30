"""Composition tests for explicit KIS and TRAKE service wiring."""

from __future__ import annotations

from typing import Any, cast

from hcmai.orchestration.pipeline import SearchService


class _Corpus:
    """Placeholder canonical Corpus dependency used only for composition."""

    def __len__(self) -> int:
        """Expose the public Corpus cardinality protocol."""

        return 0


class _Retrieval:
    """Placeholder retrieval dependency used only for composition."""


def test_kis_and_trake_share_one_temporal_service() -> None:
    """Build both explicit workflows over exactly one temporal facade."""

    service = SearchService(
        corpus=cast(Any, _Corpus()),
        retrieval=cast(Any, _Retrieval()),
    )

    assert service.kis.temporal is service.trake.temporal
    assert not hasattr(service, "pipeline_registry")


def test_health_uses_loaded_dependencies_not_task_registration() -> None:
    """Report explicit task readiness without a generic query-type registry."""

    ready = SearchService(
        corpus=cast(Any, _Corpus()),
        retrieval=cast(Any, _Retrieval()),
    ).health()["capabilities"]
    unavailable = SearchService(corpus=None, retrieval=None).health()["capabilities"]

    assert (ready["search"], ready["kis"], ready["trake"]) == (True, True, True)
    assert (unavailable["search"], unavailable["kis"], unavailable["trake"]) == (
        False,
        False,
        False,
    )
    assert "query_types" not in ready
