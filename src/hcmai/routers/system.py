"""System health and capability routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from hcmai.common.schemas import RetrievalSource
from hcmai.routers.search import StandaloneSearchDispatcher


def create_system_router(
    engine_container: dict[str, Any],
    provider_container: dict[str, Any],
    dispatcher: StandaloneSearchDispatcher,
) -> APIRouter:
    """Create health routes over the shared application runtime."""

    router = APIRouter()

    @router.get("/health")
    async def health_check() -> dict[str, Any]:
        engine = engine_container["engine"]
        frame_store = getattr(engine, "frame_store", None)
        retriever = getattr(engine, "retriever", None)
        evidence_stores = getattr(engine, "evidence_stores", {})
        store_loaded = frame_store is not None
        retriever_loaded = retriever is not None
        total_frames = (
            len(frame_store._records)
            if store_loaded and hasattr(frame_store, "_records")
            else 0
        )
        return {
            "status": "ok",
            "ready": store_loaded and retriever_loaded,
            "frame_store_loaded": store_loaded,
            "retriever_loaded": retriever_loaded,
            "total_frames": total_frames,
            "evidence_stores": {
                source.value: source in evidence_stores
                for source in (
                    RetrievalSource.CAPTION,
                    RetrievalSource.OCR,
                    RetrievalSource.ASR,
                )
            },
            "capabilities": {
                "search": retriever_loaded,
                "kisc": provider_container["kisc_agent"] is not None,
                "frame_assets": frame_store is not None,
                "query_types": dispatcher.capabilities(retriever_loaded),
            },
            "startup_messages": engine_container["startup_messages"],
        }

    return router
