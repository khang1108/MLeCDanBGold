"""Canonical frame metadata, asset, neighbor, and submission routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from hcmai.common.schemas import FrameRecord, SubmissionResult
from hcmai.orchestration.pipeline import SearchServiceUnavailableError


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
        value = frame.thumbnail_path if thumbnail else frame.image_path
        if thumbnail and value is None:
            value = frame.image_path
        path = Path(value).expanduser()
        resolved = path.resolve() if path.is_absolute() else (dataset_root / path).resolve()
        if not resolved.is_relative_to(dataset_root) or not resolved.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Frame asset not available",
            )
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
        window_ms: int = Query(default=5_000, ge=0, le=60_000),
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
