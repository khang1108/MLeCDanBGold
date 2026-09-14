"""Tests for the pure online-health report builder."""

from types import SimpleNamespace
import unittest

from hcmai.common.config import SearchConfig
from hcmai.orchestration.health import build_health_report
from hcmai.retrieval.models import RetrievalSource


class _FrameAssetStatus:
    """Small canonical asset-status fake."""

    def as_dict(self) -> dict[str, int | bool]:
        """Return a representative available asset sample."""
        return {"ready": True, "checked": 2, "available": 2, "missing": 0}


class _Corpus:
    """Minimal read-only corpus capability fake."""

    def __len__(self) -> int:
        """Return the number of canonical frames."""
        return 2

    def has_evidence(self, source: RetrievalSource) -> bool:
        """Expose source-specific evidence availability."""
        return source is not RetrievalSource.ASR

    def frame_asset_status(self) -> _FrameAssetStatus:
        """Provide deterministic frame-asset diagnostics."""
        return _FrameAssetStatus()


class HealthReportTest(unittest.TestCase):
    """Preserve health payload semantics outside SearchService."""

    def test_reports_explicit_runtime_capabilities(self) -> None:
        evidence = SimpleNamespace(
            dense=object(),
            bm25=object(),
            visual_dense_ready=True,
            context_dense_ready=False,
            asr_dense_ready=False,
        )
        service = SimpleNamespace(
            corpus=_Corpus(),
            retrieval=SimpleNamespace(active_sources=(RetrievalSource.VISUAL,)),
            config=SearchConfig(),
            llm=None,
            temporal_evidence=evidence,
            image_search=object(),
            query_preparation=object(),
            literal_text=SimpleNamespace(available_sources=(RetrievalSource.CAPTION,)),
        )

        report = build_health_report(service, startup_messages=("optional ASR unavailable",))

        self.assertTrue(report["ready"])
        self.assertEqual(report["total_frames"], 2)
        self.assertEqual(report["evidence_stores"], {
            "caption": True,
            "ocr": True,
            "asr": False,
        })
        self.assertTrue(report["capabilities"]["search"])
        self.assertTrue(report["capabilities"]["image_search"])
        self.assertTrue(report["capabilities"]["hybrid_temporal"])
        self.assertTrue(report["capabilities"]["frame_assets"])
        self.assertEqual(report["startup_messages"], ["optional ASR unavailable"])


if __name__ == "__main__":
    unittest.main()
