"""Canonical grouped-package import regression tests."""

from importlib import import_module
from importlib.util import find_spec

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "hcmai.retrieval.embedding.pipeline",
        "hcmai.retrieval.retriever.pipeline",
        "hcmai.retrieval.reranking.pipeline",
        "offline.enrichment.pipeline",
        "offline.enrichment.transcripts.pipeline",
        "hcmai.temporal.dp",
        "hcmai.temporal.planner",
        "hcmai.orchestration.temporal_search",
        "hcmai.orchestration.workflows.kis",
        "hcmai.orchestration.workflows.trake",
    ],
)
def test_canonical_grouped_import_paths_are_available(module_name: str) -> None:
    """Require production and tests to use the grouped package layout."""

    assert import_module(module_name) is not None


@pytest.mark.parametrize(
    "module_name",
    [
        "hcmai.common.schemas.vqa",
        "hcmai.common.schemas.search",
        "hcmai.common.schemas.trake",
        "hcmai.pipelines",
        "hcmai.retrieval.retriever.filtered",
        "hcmai.temporal.aligners.monotonic_dp",
        "hcmai.temporal.settings",
        "hcmai.temporal.service",
        "thundercompute.adapters.vqa",
    ],
)
def test_retired_package_paths_are_unavailable(module_name: str) -> None:
    """Keep retired modules outside the discoverable package boundary."""

    try:
        spec = find_spec(module_name)
    except ModuleNotFoundError:
        # Removing a whole legacy package makes its descendant unresolvable
        # before ``find_spec`` can return the expected ``None``.
        spec = None
    assert spec is None
    with pytest.raises(ModuleNotFoundError):
        import_module(module_name)
