"""System health and capability routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter


def create_system_router(service_container: dict[str, Any]) -> APIRouter:
    """Create health routes over the public SearchService facade."""

    router = APIRouter()

    @router.get("/health")
    async def health_check() -> dict[str, Any]:
        service = service_container.get("service")
        if service is None:
            return {
                "status": "ok",
                "ready": False,
                "frame_store_loaded": False,
                "retriever_loaded": False,
                "total_frames": 0,
                "evidence_stores": {
                    "caption": False,
                    "ocr": False,
                    "asr": False,
                },
                "capabilities": {
                    "search": False,
                    "frame_assets": False,
                    "query_types": {
                        "kis": False,
                        "vkis": False,
                        "vqa": False,
                        "trake": False,
                    },
                },
                "startup_messages": [],
            }
        return service.health(service_container.get("startup_messages", ()))

    return router
