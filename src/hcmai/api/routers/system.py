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
            payload = {
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
                    "kis": False,
                    "trake": False,
                    "frame_assets": False,
                    "frame_asset_status": {
                        "ready": False,
                        "checked": 0,
                        "available": 0,
                        "missing": 0,
                    },
                },
                "startup_messages": [],
            }
        else:
            payload = service.health(
                service_container.get("startup_messages", ())
            )

        filter_service = service_container.get("filter_service")
        filter_health = (
            filter_service.health()
            if filter_service is not None
            else {
                "ready": False,
                "catalog_version": None,
                "frame_count": 0,
            }
        )
        capabilities = payload.setdefault("capabilities", {})
        capabilities["filter"] = bool(filter_health["ready"])
        payload["filter_catalog"] = filter_health
        return payload

    return router
