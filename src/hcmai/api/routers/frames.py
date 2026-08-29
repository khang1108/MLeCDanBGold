"""Canonical frame metadata, asset, neighbor, and submission routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from hcmai.common.schemas import FrameCatalogEntry, FrameRecord, SubmissionResult
from hcmai.data.assets import FrameAssetError, FrameAssetResolver
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchServiceUnavailableError
from hcmai.common.utils.logging import get_logger


logger = get_logger(__name__)


def _search_service(container: dict[str, Any]) -> Any:
    service = container["service"]
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not initialized",
        )
    return service


def create_frames_router(
    service_container: dict[str, Any],
    dataset_root: Path,
) -> APIRouter:
    """Create routes that materialize only canonical frame identities."""

    router = APIRouter()
    fallback_resolver = FrameAssetResolver(dataset_root)

    @router.get("/api/v1/frames/{frame_id}", response_model=FrameRecord)
    async def get_frame(frame_id: str) -> FrameRecord:
        try:
            return _search_service(service_container).get_frame(frame_id)
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    @router.get("/api/v1/list-frames", response_model=list[FrameCatalogEntry])
    async def list_frames() -> list[FrameCatalogEntry]:
        """Return every canonical frame with loaded catalog evidence."""

        try:
            service = _search_service(service_container)
            return await run_in_threadpool(service.list_frames)
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error

    def frame_asset(frame_id: str, *, thumbnail: bool) -> FileResponse:
        try:
            frame = _search_service(service_container).get_frame(frame_id)
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        data = getattr(_search_service(service_container), "data", None)
        try:
            resolved = (
                data.resolve_frame_asset(frame, thumbnail=thumbnail)
                if isinstance(data, DataService)
                else fallback_resolver.resolve_frame(frame, thumbnail=thumbnail)
            )
        except (FrameAssetError, RuntimeError) as error:
            logger.warning(
                "Frame asset unavailable frame_id=%s asset=%s thumbnail=%s "
                "error_type=%s error=%s",
                frame_id,
                getattr(frame, "image_path", "<unknown>"),
                thumbnail,
                type(error).__name__,
                error,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Frame asset not available",
            ) from None
        return FileResponse(resolved)

    @router.get("/api/v1/frames/{frame_id}/thumbnail")
    async def get_frame_thumbnail(frame_id: str) -> FileResponse:
        return frame_asset(frame_id, thumbnail=True)

    @router.get("/api/v1/frames/{frame_id}/image")
    async def get_frame_image(frame_id: str) -> FileResponse:
        return frame_asset(frame_id, thumbnail=False)

    @router.get(
        "/api/v1/frames/{frame_id}/neighbors",
        response_model=list[FrameRecord],
    )
    async def get_frame_neighbors(
        frame_id: str,
        window_ms: int = Query(default=5_000, ge=0, le=3_600_000),
    ) -> list[FrameRecord]:
        try:
            return _search_service(service_container).neighbors(
                frame_id,
                window_ms=window_ms,
                include_self=True,
            )
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    @router.post("/api/v1/submit", response_model=SubmissionResult)
    async def submit_frame(frame_id: str) -> SubmissionResult:
        try:
            return _search_service(service_container).submission(frame_id)
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    return router
