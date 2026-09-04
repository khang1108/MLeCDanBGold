"""Canonical frame metadata, keyframe asset, and submission routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from hcmai.api.contracts import FrameInspectionResponse, SubmissionResult
from hcmai.corpus.models import Frame
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


def create_frames_router(service_container: dict[str, Any]) -> APIRouter:
    """Create metadata, keyframe asset, and submission routes."""

    router = APIRouter()

    @router.get("/api/v1/frames/resolve", response_model=FrameInspectionResponse)
    async def resolve_frame_at_timestamp(
        video_id: Annotated[str, Query(min_length=1)],
        timestamp_ms: Annotated[int, Query(ge=0)],
    ) -> FrameInspectionResponse:
        """Resolve canonical frame evidence for one manually opened video time."""

        try:
            return _search_service(service_container).inspect_frame_at_timestamp(
                video_id,
                timestamp_ms,
            )
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except (KeyError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    @router.get("/api/v1/frames/{frame_id}", response_model=Frame)
    async def get_frame(frame_id: str) -> Frame:
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

    def keyframe_asset(frame_id: str) -> FileResponse:
        """Resolve one canonical image without exposing its filesystem path."""

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
        corpus = getattr(_search_service(service_container), "corpus", None)
        try:
            if corpus is None:
                raise RuntimeError("Frame store not loaded")
            resolved = corpus.image_path(frame_id)
        except (OSError, RuntimeError) as error:
            logger.warning(
                "Keyframe asset unavailable frame_id=%s asset=%s "
                "error_type=%s error=%s",
                frame_id,
                getattr(frame, "image_path", "<unknown>"),
                type(error).__name__,
                error,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Frame asset not available",
            ) from None
        return FileResponse(resolved)

    @router.get("/api/v1/keyframes/{frame_id}")
    async def get_keyframe(frame_id: str) -> FileResponse:
        """Serve the canonical keyframe image for one internal frame ID."""

        return keyframe_asset(frame_id)

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
