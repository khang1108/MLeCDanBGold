"""Canonical grouped-package import regression tests."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "hcmai.retrieval.embedding.pipeline",
        "hcmai.retrieval.retriever.pipeline",
        "hcmai.retrieval.reranking.pipeline",
        "hcmai.pipelines.trake",
        "hcmai.data.enrichment.pipeline",
        "hcmai.data.enrichment.transcripts.pipeline",
        "hcmai.orchestration.workflows.kis",
        "hcmai.orchestration.workflows.trake",
    ],
)
def test_canonical_grouped_import_paths_are_available(module_name: str) -> None:
    """Require production and tests to use the grouped package layout."""

    assert import_module(module_name) is not None
