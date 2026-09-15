"""Tests for the pure online-health report builder."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from hcmai.common.config import SearchConfig
from hcmai.orchestration.utils.health import build_health_report
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


def _service(*, intent_resolver, event_translator, temporal_evidence):
    """Build the smallest service-shaped fixture for KIS readiness tests."""
    return SimpleNamespace(
        intent_resolver=intent_resolver,
        event_translator=event_translator,
        temporal_evidence=temporal_evidence,
    )


def test_kis_not_ready_when_intent_resolver_is_missing() -> None:
    """KIS readiness must require the initial semantic resolver capability."""
    service = _service(
        intent_resolver=None,
        event_translator=object(),
        temporal_evidence=object(),
    )

    report = build_health_report(service)

    assert report["capabilities"]["intent_resolution"] is False
    assert report["capabilities"]["retrieval"] is True
    assert report["capabilities"]["event_translation"] is True
    assert report["capabilities"]["kis"] is False


def test_kis_ready_with_resolver_and_temporal_retrieval() -> None:
    """KIS becomes ready when retrieval and intent resolution are present."""
    service = _service(
        intent_resolver=object(),
        event_translator=object(),
        temporal_evidence=object(),
    )

    report = build_health_report(service)

    assert report["capabilities"]["intent_resolution"] is True
    assert report["capabilities"]["retrieval"] is True
    assert report["capabilities"]["kis"] is True


def test_kis_readiness_does_not_require_event_translation() -> None:
    """English and image-only paths remain KIS-ready without translation."""
    service = _service(
        intent_resolver=object(),
        event_translator=None,
        temporal_evidence=object(),
    )

    report = build_health_report(service)

    assert report["capabilities"]["event_translation"] is False
    assert report["capabilities"]["kis"] is True


def test_health_report_does_not_call_configured_provider_health_methods() -> None:
    """Health projection must remain observational when an LLM is configured."""
    llm = Mock()
    llm.gateway_health.side_effect = AssertionError("gateway health was called")
    llm.capability_health.side_effect = AssertionError("capability health was called")
    service = SimpleNamespace(
        corpus=_Corpus(),
        retrieval=SimpleNamespace(active_sources=(RetrievalSource.VISUAL,)),
        config=SearchConfig(),
        llm=llm,
        temporal_evidence=object(),
        image_search=None,
        event_translator=None,
        intent_resolver=object(),
        literal_text=None,
    )

    report = build_health_report(service)

    assert report["capabilities"]["kis"] is True
    llm.gateway_health.assert_not_called()
    llm.capability_health.assert_not_called()


def test_retrieval_and_kis_not_ready_without_temporal_evidence() -> None:
    """Retrieval readiness requires temporal evidence even with a corpus."""
    service = SimpleNamespace(
        corpus=_Corpus(),
        retrieval=SimpleNamespace(active_sources=(RetrievalSource.VISUAL,)),
        config=SearchConfig(),
        llm=None,
        temporal_evidence=None,
        image_search=None,
        event_translator=object(),
        intent_resolver=object(),
        literal_text=None,
    )

    report = build_health_report(service)

    assert report["capabilities"]["retrieval"] is False
    assert report["capabilities"]["kis"] is False


def test_retrieval_and_kis_not_ready_without_corpus() -> None:
    """Retrieval readiness requires canonical corpus data."""
    service = SimpleNamespace(
        corpus=None,
        retrieval=SimpleNamespace(active_sources=(RetrievalSource.VISUAL,)),
        config=SearchConfig(),
        llm=None,
        temporal_evidence=object(),
        image_search=None,
        event_translator=object(),
        intent_resolver=object(),
        literal_text=None,
    )

    report = build_health_report(service)

    assert report["capabilities"]["retrieval"] is False
    assert report["capabilities"]["kis"] is False


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
            event_translator=object(),
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
        self.assertTrue(report["capabilities"]["event_translation"])
        self.assertTrue(report["capabilities"]["frame_assets"])
        self.assertEqual(report["startup_messages"], ["optional ASR unavailable"])


if __name__ == "__main__":
    unittest.main()
