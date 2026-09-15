"""HTTP router for revisioned KIS multimodal video search."""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, File, Header, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts.kis import (
    KISRevisionSearchRequest,
    KISRevisionSearchResponse,
)
from hcmai.common.utils.logging import get_logger
from hcmai.inference.errors import InferenceResponseError, InferenceUnavailableError
from hcmai.kis.assets import InvalidImageError
from hcmai.kis.models import KISImageRef
from hcmai.kis.resolver import KISResolutionError
from hcmai.orchestration.utils.errors import InvalidQueryInputError, RevisionConflictError
from hcmai.orchestration.pipeline import SearchServiceUnavailableError
from hcmai.retrieval.translation.service import EventTranslationError
from hcmai.vbs.models import ApiClientAnswer, QueryEvent, QueryResultLog, RankedAnswer

logger = get_logger(__name__)


def create_kis_router(service_container: dict[str, Any]) -> APIRouter:
    """Create the revisioned KIS search router."""
    router = APIRouter()

    @router.post("/api/v1/kis/search", response_model=KISRevisionSearchResponse)
    async def search_kis_revision(
        request: KISRevisionSearchRequest,
        response: Response,
        user_id: Annotated[str | None, Header(alias="X-VBS-User-ID")] = None,
    ) -> KISRevisionSearchResponse:
        """Execute a revisioned KIS search with semantic intent resolution."""
        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )

        try:
            result = await run_in_threadpool(service.search_kis_revision, request)
            await _record_kis_dres_log(
                service_container,
                response,
                user_id=user_id,
                query_text=result.intent.query_text,
                results=result.results,
            )
            return result
        except RevisionConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error
        except InvalidQueryInputError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        except (KISResolutionError, EventTranslationError, InferenceResponseError) as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(error),
            ) from error
        except (InferenceUnavailableError, SearchServiceUnavailableError) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except Exception as error:
            logger.exception("KIS revision search failed: %s", error)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Inference error: {error}",
            ) from error

    @router.post("/api/v1/kis/assets/images", response_model=KISImageRef)
    async def upload_kis_image(file: UploadFile = File(...)) -> KISImageRef:
        """Store a validated user-supplied query image and return its stable ref.

        The returned ``asset_id`` can be attached to KIS events as an image
        reference. The canonical copy is available via the paired GET route.
        """
        service = service_container.get("service")
        if service is None or service.kis_image_assets is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="KIS image asset store not available",
            )
        payload = await file.read(service.api_config.image_max_upload_bytes + 1)
        try:
            return await run_in_threadpool(
                service.kis_image_assets.put,
                payload,
                file.content_type,
            )
        except InvalidImageError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

    @router.get("/api/v1/kis/assets/images/{asset_id}")
    async def get_kis_image(asset_id: str) -> Response:
        """Return the validated original bytes for a previously uploaded image.

        Responses are cache-immutable because asset IDs are content-addressed:
        the same ID always returns the same bytes.
        """
        service = service_container.get("service")
        if service is None or service.kis_image_assets is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="KIS image asset store not available",
            )
        try:
            payload, content_type = await run_in_threadpool(
                service.kis_image_assets.read, asset_id
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset not found: {asset_id}",
            ) from error
        return Response(
            content=payload,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    return router


async def _record_kis_dres_log(
    service_container: dict[str, Any],
    response: Response,
    *,
    user_id: str | None,
    query_text: str,
    results: list[Any],
) -> None:
    """Log search results to DRES server with the canonical intent query text."""
    log_status = "skipped"
    try:
        vbs_service = service_container.get("vbs_service")
        if (
            user_id is not None
            and user_id.strip()
            and vbs_service is not None
            and vbs_service.session_status(user_id).get("connected")
        ):
            evaluation_id = await vbs_service.resolve_evaluation(user_id)
            timestamp = _now_ms()
            ranked = [
                RankedAnswer(
                    rank=index + 1,
                    answer=ApiClientAnswer(
                        media_item_name=vbs_service.media_item_name(result.video_id),
                        start=result.timestamp_ms,
                        end=result.timestamp_ms,
                    ),
                )
                for index, result in enumerate(results)
            ]
            payload = QueryResultLog(
                timestamp=timestamp,
                sort_type="list",
                result_set_availability="",
                results=ranked,
                events=[
                    QueryEvent(
                        timestamp=timestamp,
                        category="TEXT",
                        event_type="SEARCH",
                        value=query_text,
                    )
                ],
            )
            await vbs_service.log_results(user_id, evaluation_id, payload)
            log_status = "sent"
    except Exception:
        logger.warning("DRES result logging failed")
        log_status = "failed"
    response.headers["X-DRES-Log-Status"] = log_status


def _now_ms() -> int:
    return int(time.time() * 1000)
