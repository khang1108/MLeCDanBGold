"""Standalone competition-task search routing."""

from __future__ import annotations

import hashlib
import json
import time
from time import perf_counter
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Header, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts import (
    FilterRequest,
    FilterResponse,
    ImageSearchResponse,
)
from hcmai.api.result_logging import record_dres_result_log
from hcmai.common.utils.logging import get_logger
from hcmai.orchestration.pipeline import SearchServiceUnavailableError
from hcmai.orchestration.utils.errors import SearchServiceGatewayError
from hcmai.orchestration.workflows.image_search import (
    ImageQueryTooLargeError,
    InvalidImageQueryError,
)

logger = get_logger(__name__)


def create_search_router(service_container: dict[str, Any]) -> APIRouter:
    """Create the standalone frame-search HTTP router."""

    router = APIRouter()

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
        http_response: Response,
        top_k: Annotated[int, Form(ge=1, le=100)] = 20,
        user_id: Annotated[str | None, Header(alias="X-VBS-User-ID")] = None,
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
            result = await run_in_threadpool(
                service.search_image,
                payload,
                content_type=image.content_type,
                top_k=top_k,
            )
            await record_dres_result_log(
                service_container,
                http_response,
                user_id=user_id,
                category="IMAGE",
                event_value=(
                    f"filename={image.filename or ''};"
                    f"sha256={hashlib.sha256(payload).hexdigest()}"
                ),
                results=result.results,
                now_fn=_now_ms,
            )
            return result
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
        except SearchServiceGatewayError as error:
            logger.error("API image search upstream gateway error=%s", error)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
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

    @router.post("/api/v1/filter", response_model=FilterResponse)
    async def filter_frames(
        request: FilterRequest,
        http_response: Response,
        user_id: Annotated[str | None, Header(alias="X-VBS-User-ID")] = None,
    ) -> FilterResponse:
        """Run direct substring matching over evidence loaded at startup."""

        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        started = perf_counter()
        try:
            result = await run_in_threadpool(service.filter_frames, request)
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        logger.info(
            "Literal filter completed matches=%d page=%d elapsed_ms=%.1f",
            result.total_results,
            result.page_id,
            (perf_counter() - started) * 1_000,
        )
        predicates = {
            "metadata_filters": request.metadata_filters.model_dump(
                mode="json",
                exclude_none=True,
            ),
        }
        if request.folder_id is not None:
            predicates["folder_id"] = request.folder_id
        if request.video_id is not None:
            predicates["video_id"] = request.video_id
        await record_dres_result_log(
            service_container,
            http_response,
            user_id=user_id,
            category="FILTER",
            event_value=json.dumps(
                predicates,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            results=result.results,
            rank_offset=(result.page_id - 1) * result.frames_per_pages,
            now_fn=_now_ms,
        )
        return result

    return router


def _now_ms() -> int:
    """Return the wall-clock Unix epoch used by DRES event and result logs."""
    return int(time.time() * 1000)
