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
) -> dict[str, Any]:
    """Build readiness and capability state from composed dependencies.

    The search service is already assembled before this function runs. This
    report only observes its dependencies, so readiness checks cannot trigger
    model loading, index discovery, or request-time retrieval work.
    """
    corpus = getattr(service, "corpus", None)
    retrieval = getattr(service, "retrieval", None)
    temporal_evidence = getattr(service, "temporal_evidence", None)

    # Minimal health fixtures may expose only the capabilities under test. A
    # real SearchService always declares ``corpus``, so an explicitly missing
    # corpus remains unavailable while an omitted fixture field is not treated
    # as a negative signal for semantic readiness.
    corpus_ready = (
        corpus is not None
        if hasattr(service, "corpus")
        else temporal_evidence is not None
    )
    retriever_loaded = retrieval is not None
    retrieval_ready = corpus_ready and temporal_evidence is not None
    intent_resolution_ready = getattr(service, "intent_resolver", None) is not None
    event_translation_ready = getattr(service, "event_translator", None) is not None
    kis_ready = retrieval_ready and intent_resolution_ready
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
    asset_status = _frame_asset_status(corpus)
    active_sources = (
        set(getattr(retrieval, "active_sources", (RetrievalSource.VISUAL,)))
        if retrieval is not None
        else set()
    )
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
        return corpus.frame_asset_status().as_dict()
    except (AttributeError, OSError, RuntimeError, TypeError):
        return {"ready": False, "checked": 0, "available": 0, "missing": 0}
