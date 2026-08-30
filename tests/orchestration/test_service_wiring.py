"""Composition tests for explicit KIS and TRAKE service wiring."""

from __future__ import annotations

from typing import Any, cast

from hcmai.orchestration.pipeline import SearchService


class _Corpus:
    """Placeholder canonical Corpus dependency used only for composition."""

    def __len__(self) -> int:
        """Expose the public Corpus cardinality protocol."""

        return 0

    def frame_asset_status(self) -> Any:
        """Provide the public diagnostic projection used by health checks."""

        return _AssetStatus()

    def has_evidence(self, source: Any) -> bool:
        """Expose the public evidence-availability protocol."""

        del source
        return False


class _AssetStatus:
    """Minimal public asset-health projection for composition tests."""

    def as_dict(self) -> dict[str, int | bool]:
        """Return the diagnostic shape consumed by SearchService.health."""

        return {"ready": False, "checked": 0, "available": 0, "missing": 0}


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
