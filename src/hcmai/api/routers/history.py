"""Query replay history and viewed-frame routes.

This router owns lossless KIS/legacy replay snapshots and viewing activity.
It intentionally has no submission-file or answer-workspace endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool

from hcmai.api.contracts.history import (
    QueryHistoryCreate,
    QueryHistoryList,
    QueryHistoryRecord,
    QueryHistoryViewedFrameUpdate,
)
from hcmai.api.history import WorkspaceStore
from hcmai.orchestration.pipeline import SearchServiceUnavailableError


def create_history_router(service_container: dict[str, Any]) -> APIRouter:
    """Expose existing query-history persistence without submission coupling."""

    router = APIRouter()

    def _store() -> WorkspaceStore:
        store = service_container.get("workspace_store")
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Query history store is not configured",
            )
        return store

    @router.post(
        "/api/v1/query-history",
        response_model=QueryHistoryRecord,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_query_history(data: QueryHistoryCreate) -> QueryHistoryRecord:
        """Persist a successful search snapshot without rewriting its evidence."""

        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        try:
            _validate_snapshot_shape(data.result_snapshot)
            _validate_snapshot_frames(service, data.result_snapshot)
            return await run_in_threadpool(_store().create_history, data)
        except HTTPException:
            raise
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
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except Exception as error:
            if "UNIQUE constraint failed" in str(error):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Query history already exists",
                ) from error
            raise

    @router.get("/api/v1/query-history", response_model=QueryHistoryList)
    async def get_query_history(
        user_id: str = Query(min_length=1),
    ) -> QueryHistoryList:
        """Return the newest stored searches for one participant."""

        return QueryHistoryList(
            items=await run_in_threadpool(_store().get_recent_history, user_id)
        )

    @router.patch(
        "/api/v1/query-history/{query_id}/viewed-frame",
        response_model=QueryHistoryRecord,
    )
    async def update_viewed_frame(
        query_id: str,
        data: QueryHistoryViewedFrameUpdate,
    ) -> QueryHistoryRecord:
        """Record a canonical frame opened from one replay snapshot."""

        try:
            service = service_container.get("service")
            if service is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Search service not initialized",
                )
            service.get_frame(data.frame_id)
            return await run_in_threadpool(
                _store().update_viewed_frame,
                query_id,
                data.frame_id,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    return router


def _validate_snapshot_frames(service: Any, snapshot: dict[str, Any]) -> None:
    """Check canonical frame references when a replay snapshot contains them."""

    results = snapshot.get("results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict) or not isinstance(result.get("frame_id"), str):
                raise ValueError("KIS history results require canonical frame_id values")
            frame = service.get_frame(result["frame_id"])
            for field in ("video_id", "frame_idx", "timestamp_ms"):
                if field in result and result[field] != getattr(frame, field):
                    raise ValueError(
                        f"KIS history result {field} does not match canonical frame metadata"
                    )

    paths = snapshot.get("paths")
    if isinstance(paths, list):
        for path in paths:
            if not isinstance(path, dict) or not isinstance(path.get("frame_ids"), list):
                raise ValueError("TRAKE history paths require canonical frame_ids arrays")
            video_id = path.get("video_id")
            for frame_id in path["frame_ids"]:
                if not isinstance(frame_id, str):
                    raise ValueError("TRAKE history frame IDs must be strings")
                frame = service.get_frame(frame_id)
                if video_id is not None and frame.video_id != video_id:
                    raise ValueError(
                        "TRAKE history path frames must belong to its declared video_id"
                    )


def _validate_snapshot_shape(snapshot: dict[str, Any]) -> None:
    """Require one supported replay result collection without narrowing its data."""

    if not isinstance(snapshot.get("results"), list) and not isinstance(snapshot.get("paths"), list):
        raise ValueError("History snapshot must contain KIS results or legacy TRAKE paths")


__all__ = ["create_history_router"]
