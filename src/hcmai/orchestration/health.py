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
    corpus = service.corpus
    retrieval = service.retrieval
    temporal_evidence = service.temporal_evidence

    corpus_ready = corpus is not None
    retrieval_ready = retrieval is not None
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
    capability_health = (
        getattr(service.llm, "capability_health", None)
        if service.llm is not None
        else None
    )
    remote_capabilities = (
        capability_health()
        if capability_health is not None
        else {
            "embedding": False,
            "reranking": False,
            "structured_parsing": False,
        }
    )
    search_ready = corpus_ready and temporal_evidence is not None

    return {
        "status": "ok",
        "ready": corpus_ready and retrieval_ready,
        "frame_store_loaded": corpus_ready,
        "retriever_loaded": retrieval_ready,
        "total_frames": len(corpus) if corpus is not None else 0,
        "evidence_stores": {
            source.value: corpus.has_evidence(source) if corpus is not None else False
            for source in (
                RetrievalSource.CAPTION,
                RetrievalSource.OCR,
                RetrievalSource.ASR,
            )
        },
        "remote_inference": (
            service.llm.gateway_health()
            if service.llm is not None
            else {
                "configured": False,
                "circuit_state": "not_configured",
            }
        ),
        "retrieval_modalities": {
            source.value: {
                "active": source in active_sources,
                "required": source in service.config.fusion.required_sources,
            }
            for source in RetrievalSource
        },
        "observability": METRICS.snapshot(),
        "capabilities": {
            "search": search_ready,
            "image_search": service.image_search is not None,
            "kis": search_ready,
            "trake": search_ready,
            "shared_retrieval": retrieval_ready,
            "visual_dense": visual_dense_ready,
            "context_dense": context_dense_ready,
            "asr_dense": asr_dense_ready,
            "dense_temporal": dense_temporal_ready,
            "bm25": bm25_ready,
            "hybrid_temporal": dense_temporal_ready and bm25_ready,
            "query_preparation": service.query_preparation is not None,
            "filter": bool(
                service.literal_text is not None
                and service.literal_text.available_sources
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
    except (OSError, RuntimeError):
        return {"ready": False, "checked": 0, "available": 0, "missing": 0}
