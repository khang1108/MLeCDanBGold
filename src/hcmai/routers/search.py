"""Standalone competition-task search routing."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, HTTPException, status

from hcmai.common.schemas import SearchRequest, SearchResponse, TaskType
from hcmai.common.utils.logging import get_logger

logger = get_logger(__name__)


class _UnsupportedSearchTaskError(ValueError):
    """Raised when a non-standalone task reaches this dispatcher."""


class _UnavailableSearchPipelineError(RuntimeError):
    """Raised when a known standalone task has no executable pipeline."""


class SearchEngineUnavailableError(RuntimeError):
    """Raised when the shared frame-search pipeline is not ready."""


class StandaloneSearchDispatcher:
    """Dispatch standalone task types to their configured pipelines."""

    def __init__(
        self,
        frame_search: Callable[[SearchRequest], SearchResponse],
    ) -> None:
        self._pipelines: dict[
            TaskType, Callable[[SearchRequest], SearchResponse] | None
        ] = {
            TaskType.KIS: frame_search,
            TaskType.VKIS: frame_search,
            TaskType.VQA: None,
            TaskType.TRAKE: None,
        }

    def dispatch(self, request: SearchRequest) -> SearchResponse:
        if request.session_id is not None or request.feedback is not None:
            raise _UnsupportedSearchTaskError(
                "session_id and feedback are not supported by standalone search"
            )
        if request.query_type not in self._pipelines:
            raise _UnsupportedSearchTaskError(
                f"query_type {request.query_type.value!r} is not a "
                "standalone search task"
            )
        pipeline = self._pipelines[request.query_type]
        if pipeline is None:
            raise _UnavailableSearchPipelineError(
                f"pipeline for query_type {request.query_type.value!r} "
                "is not available"
            )
        logger.info(
            "Routing standalone query_type=%s pipeline=frame_search",
            request.query_type.value,
        )
        return pipeline(request)

    def capabilities(self, ready: bool) -> dict[str, bool]:
        return {
            task.value: pipeline is not None and ready
            for task, pipeline in self._pipelines.items()
        }


def create_search_router(
    dispatcher: StandaloneSearchDispatcher,
) -> APIRouter:
    """Create the standalone frame-search HTTP router."""

    router = APIRouter()

    @router.post("/api/v1/search", response_model=SearchResponse)
    async def search_frames(request: SearchRequest) -> SearchResponse:
        try:
            return dispatcher.dispatch(request)
        except _UnsupportedSearchTaskError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        except _UnavailableSearchPipelineError as error:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=str(error),
            ) from error
        except SearchEngineUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except KeyError as error:
            logger.warning("API search request failed error=%s", error)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except Exception:
            logger.exception("API search request failed unexpectedly")
            raise

    return router
