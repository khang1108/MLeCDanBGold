"""Read-only health reporting for the online search runtime.

This module will own readiness and capability projection. It does not load
artifacts, mutate workflows, or perform retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hcmai.common.observability import METRICS
from hcmai.retrieval.models import RetrievalSource


def build_health_report(
    service: Any,
    *,
    startup_messages: Sequence[str] = (),
    remote_status: Any | None = None,
) -> dict[str, Any]:
    """Build readiness and capability state from composed dependencies.

    The search service is already assembled before this function runs. This
    report only observes its dependencies, so readiness checks cannot trigger
    model loading, index discovery, or request-time retrieval work.
    """
    corpus = getattr(service, "corpus", None)
    retrieval = getattr(service, "retrieval", None)
    temporal_evidence = getattr(service, "temporal_evidence", None)
    remote_client = getattr(service, "remote_retrieval", None)

    if remote_status is None and remote_client is not None:
        try:
            remote_status = remote_client.probe()
        except Exception:
            remote_status = None

    if remote_status is not None:
        remote_report = {
            "configured": True,
            "reachable": bool(remote_status.reachable),
            "ready": bool(remote_status.ready),
            "target": remote_status.target,
            "scoring_revision": remote_status.scoring_revision,
            "active_modalities": list(remote_status.active_modalities),
        }
    elif remote_client is not None:
        remote_report = {
            "configured": True,
            "reachable": False,
            "ready": False,
            "target": getattr(getattr(remote_client, "settings", None), "target", None),
            "scoring_revision": None,
            "active_modalities": [],
        }
    else:
        remote_report = {
            "configured": False,
            "reachable": False,
            "ready": False,
            "target": None,
            "scoring_revision": None,
            "active_modalities": [],
        }

    # Minimal health fixtures may expose only the capabilities under test. A
    # real SearchService always declares ``corpus``, so an explicitly missing
    # corpus remains unavailable while an omitted fixture field is not treated
    # as a negative signal for semantic readiness.
    corpus_ready = (
        corpus is not None
        if hasattr(service, "corpus")
        else (temporal_evidence is not None or remote_report["ready"])
    )

    if remote_client is not None:
        retriever_loaded = remote_report["ready"]
        retrieval_ready = corpus_ready and retriever_loaded
        active_mods = set(remote_report["active_modalities"])
        visual_dense_ready = "visual" in active_mods
        context_dense_ready = "context" in active_mods
        asr_dense_ready = "asr" in active_mods
        dense_temporal_ready = visual_dense_ready or context_dense_ready or asr_dense_ready
        bm25_ready = "bm25" in active_mods
        active_sources = set()
        if "visual" in active_mods:
            active_sources.add(RetrievalSource.VISUAL)
        if "context" in active_mods:
            active_sources.add(RetrievalSource.CONTEXT)
            active_sources.add(RetrievalSource.CAPTION)
            active_sources.add(RetrievalSource.OCR)
        if "asr" in active_mods:
            active_sources.add(RetrievalSource.ASR)
    else:
        retriever_loaded = retrieval is not None
        retrieval_ready = corpus_ready and temporal_evidence is not None
        dense_temporal_ready = (
            temporal_evidence is not None
            and getattr(temporal_evidence, "dense", None) is not None
        )
        visual_dense_ready = bool(
            getattr(temporal_evidence, "visual_dense_ready", dense_temporal_ready)
        )
        context_dense_ready = bool(
            getattr(temporal_evidence, "context_dense_ready", dense_temporal_ready)
        )
        asr_dense_ready = bool(
            getattr(temporal_evidence, "asr_dense_ready", dense_temporal_ready)
        )
        bm25_ready = (
            temporal_evidence is not None
            and getattr(temporal_evidence, "bm25", None) is not None
        )
        active_sources = (
            set(getattr(retrieval, "active_sources", (RetrievalSource.VISUAL,)))
            if retrieval is not None
            else set()
        )

    intent_resolution_ready = getattr(service, "intent_resolver", None) is not None
    event_translation_ready = getattr(service, "event_translator", None) is not None
    kis_ready = retrieval_ready and intent_resolution_ready
    asset_status = _frame_asset_status(corpus)
    llm = getattr(service, "llm", None)
    # Readiness is intentionally observational. Calling provider health or
    # capability methods here can perform network/model work on a health poll.
    remote_capabilities = {
        "embedding": llm is not None,
        "reranking": llm is not None,
        "structured_parsing": llm is not None,
    }
    search_ready = retrieval_ready
    config = getattr(service, "config", None)
    fusion = getattr(config, "fusion", None)
    required_sources = getattr(fusion, "required_sources", ())

    def has_evidence(source: RetrievalSource) -> bool:
        """Read corpus evidence availability without making diagnostics fatal."""
        method = getattr(corpus, "has_evidence", None)
        return bool(corpus is not None and method is not None and method(source))

    def corpus_length() -> int:
        """Return the corpus size when the composed dependency exposes it."""
        try:
            return len(corpus) if corpus is not None else 0
        except TypeError:
            return 0

    return {
        "status": "ok",
        "ready": corpus_ready and retriever_loaded,
        "frame_store_loaded": corpus_ready,
        "retriever_loaded": retriever_loaded,
        "remote_retrieval": remote_report,
        "total_frames": corpus_length(),
        "evidence_stores": {
            source.value: has_evidence(source)
            for source in (
                RetrievalSource.CAPTION,
                RetrievalSource.OCR,
                RetrievalSource.ASR,
            )
        },
        "remote_inference": {
            "configured": llm is not None,
            "circuit_state": "configured" if llm is not None else "not_configured",
        },
        "retrieval_modalities": {
            source.value: {
                "active": source in active_sources,
                "required": source in required_sources,
            }
            for source in RetrievalSource
        },
        "observability": METRICS.snapshot(),
        "capabilities": {
            "search": search_ready,
            "image_search": getattr(service, "image_search", None) is not None,
            "kis": kis_ready,
            "trake": search_ready,
            "shared_retrieval": retrieval_ready,
            "retrieval": retrieval_ready,
            "intent_resolution": intent_resolution_ready,
            "visual_dense": visual_dense_ready,
            "context_dense": context_dense_ready,
            "asr_dense": asr_dense_ready,
            "dense_temporal": dense_temporal_ready,
            "bm25": bm25_ready,
            "hybrid_temporal": dense_temporal_ready and bm25_ready,
            "event_translation": event_translation_ready,
            "filter": bool(
                getattr(service, "literal_text", None) is not None
                and getattr(service.literal_text, "available_sources", ())
            ),
            "remote_inference": remote_capabilities,
            "frame_assets": asset_status["ready"],
            "frame_asset_status": asset_status,
        },
        "startup_messages": list(startup_messages),
    }


def _frame_asset_status(corpus: Any | None) -> dict[str, int | bool]:
    """Sample frame assets without allowing diagnostics to break serving."""
    if corpus is None:
        return {"ready": False, "checked": 0, "available": 0, "missing": 0}
    try:
        status = corpus.frame_asset_status().as_dict()
        if isinstance(status, dict):
            return status
        return {"ready": False, "checked": 0, "available": 0, "missing": 0}
    except Exception:
        return {"ready": False, "checked": 0, "available": 0, "missing": 0}
