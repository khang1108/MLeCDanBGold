"""Standalone competition-task search routing."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts import ImageSearchResponse, SearchRequest, SearchResponse
from hcmai.common.utils.logging import get_logger
from hcmai.orchestration.pipeline import SearchServiceUnavailableError
from hcmai.orchestration.workflows.image_search import (
    ImageQueryTooLargeError,
    InvalidImageQueryError,
)

logger = get_logger(__name__)


def create_search_router(service_container: dict[str, Any]) -> APIRouter:
    """Create the standalone frame-search HTTP router."""

    router = APIRouter()

    @router.post("/api/v1/search", response_model=SearchResponse)
    async def search_frames(request: SearchRequest) -> SearchResponse:
        """Validate and delegate one standalone KIS-family search request."""

        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        try:
            return await run_in_threadpool(service.search_kis, request)
        except KeyError as error:
            logger.warning("API search request failed error=%s", error)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except Exception:
            logger.exception("API search request failed unexpectedly")
            raise

    @router.post(
        "/api/v1/search/image",
        response_model=ImageSearchResponse,
        responses={
            413: {
                "description": "Encoded or decoded image exceeds configured limits"
            },
            415: {"description": "Unsupported image media type"},
            503: {"description": "Image-search dependencies are unavailable"},
        },
    )
    async def search_frames_by_image(
        image: Annotated[
            UploadFile,
            File(description="JPEG, PNG, or WebP query image"),
        ],
        top_k: Annotated[int, Form(ge=1, le=100)] = 20,
    ) -> ImageSearchResponse:
        """Search canonical keyframes using one uploaded visual query."""

        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        image_search = getattr(service, "image_search", None)
        if image_search is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Image search service not initialized",
            )
        if image.content_type not in image_search.SUPPORTED_MEDIA_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="image must use JPEG, PNG, or WebP media type",
            )

        payload = await image.read(image_search.max_upload_bytes + 1)
        try:
            return await run_in_threadpool(
                service.search_image,
                payload,
                content_type=image.content_type,
                top_k=top_k,
            )
        except ImageQueryTooLargeError as error:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(error),
            ) from error
        except InvalidImageQueryError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except KeyError as error:
            logger.warning("API image search request failed error=%s", error)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except Exception:
            logger.exception("API image search request failed unexpectedly")
            raise

    return router
