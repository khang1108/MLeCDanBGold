"""Tests for explicit online-workflow dependency composition."""

import unittest

from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.workflows.image_search import ImageSearchService


class _RetrievalThatTracksInspection:
    """Minimal retrieval fake that records deprecated constructor discovery."""

    def __init__(self) -> None:
        self.source_retriever_calls = 0

    def source_retriever(self, _source: object) -> None:
        """Record an attempt to inspect an image encoder."""
        self.source_retriever_calls += 1
        return None


class _VisualRetrieverThatTracksDiscovery:
    """Visual retriever fake that remains usable through the legacy lookup."""

    def __init__(self) -> None:
        self.source_retriever_calls = 0

    def source_retriever(self, _source: object) -> "_VisualRetrieverThatTracksDiscovery":
        """Record a deprecated request to rediscover this already-selected retriever."""
        self.source_retriever_calls += 1
        return self


class SearchServiceCompositionTest(unittest.TestCase):
    """Keep runtime discovery at setup instead of request composition."""

    def test_constructor_does_not_inspect_retrieval_for_optional_dependencies(self) -> None:
        retrieval = _RetrievalThatTracksInspection()

        service = SearchService(
            corpus=None,
            retrieval=retrieval,  # type: ignore[arg-type]
            temporal_evidence=None,
        )

        self.assertEqual(retrieval.source_retriever_calls, 0)
        self.assertIsNone(service.temporal_evidence)
        self.assertIsNone(service.image_search)

    def test_image_workflow_receives_the_selected_visual_retriever(self) -> None:
        visual = _VisualRetrieverThatTracksDiscovery()

        ImageSearchService(
            object(),  # type: ignore[arg-type]
            visual,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            max_upload_bytes=1,
            max_pixels=1,
        )

        self.assertEqual(visual.source_retriever_calls, 0)


if __name__ == "__main__":
    unittest.main()
