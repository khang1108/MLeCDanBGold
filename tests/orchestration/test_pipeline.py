"""Tests for explicit online-workflow dependency composition."""

import unittest

from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.workflows.search.image import ImageSearchService


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

    def test_materialize_resolves_visual_score(self) -> None:
        from unittest.mock import Mock
        from types import SimpleNamespace
        from hcmai.retrieval.models import RetrievalCandidate, RetrievalSource

        corpus = Mock()
        corpus.frame.return_value = SimpleNamespace(
            video_id="V001",
            frame_id="F001",
            frame_idx=10,
            timestamp_ms=1000,
            fps=25.0,
        )
        corpus.title.return_value = "Video Title"
        corpus.caption.return_value = "A test caption"
        corpus.ocr.return_value = "Some OCR"
        corpus.objects.return_value = []
        corpus.transcript.return_value = "transcript text"
        service = ImageSearchService(
            corpus,
            _VisualRetrieverThatTracksDiscovery(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            max_upload_bytes=1000,
            max_pixels=1000,
        )
        candidate = RetrievalCandidate(
            frame_id="F001",
            source_scores={RetrievalSource.VISUAL: 0.95},
        )
        result = service._materialize(candidate)
        self.assertEqual(result.video_id, "V001")
        self.assertEqual(result.frame_idx, 10)
        self.assertEqual(result.score, 0.95)



if __name__ == "__main__":
    unittest.main()
