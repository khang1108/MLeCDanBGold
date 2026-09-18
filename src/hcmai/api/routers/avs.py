"""Thin HTTP adapter for AVS direct text search."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Response, status
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts.avs import AvsSearchRequest, AvsSearchResponse
from hcmai.api.result_logging import record_dres_result_log
from hcmai.common.utils.logging import get_logger
from hcmai.orchestration.pipeline import SearchServiceUnavailableError
from hcmai.orchestration.utils.errors import (
    InvalidQueryInputError,
    SearchServiceGatewayError,
)

logger = get_logger(__name__)


def create_avs_router(service_container: dict[str, Any]) -> APIRouter:
    """Create the AVS router over the public SearchService facade."""

    router = APIRouter()

    @router.post("/api/v1/avs/search", response_model=AvsSearchResponse)
    async def search_avs(
        request: AvsSearchRequest,
        http_response: Response,
        user_id: Annotated[str | None, Header(alias="X-VBS-User-ID")] = None,
    ) -> AvsSearchResponse:
        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        try:
            result = await run_in_threadpool(service.search_avs, request)
            await record_dres_result_log(
                service_container,
                http_response,
                user_id=user_id,
                category="TEXT",
                event_value=request.query,
                results=result.results,
            )
            return result
        except InvalidQueryInputError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except SearchServiceGatewayError as error:
            logger.error("API AVS upstream gateway error=%s", error)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(error),
            ) from error
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except Exception:
            logger.exception("API AVS request failed unexpectedly")
            raise

    return router
