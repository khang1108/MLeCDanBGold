"""Placeholder HTTP endpoint for the Filter feature under development.

This module intentionally owns no matching, ranking, catalog, or artifact
logic. The stable route remains available so the frontend integration can be
developed without implying that Filter is ready for use.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status


def create_filter_router() -> APIRouter:
    """Create the stable Filter route as an explicit development placeholder."""

    router = APIRouter()

    @router.post("/api/v1/filter")
    async def filter_frames() -> None:
        """Report that Filter behavior is intentionally not implemented yet."""

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Tính năng Filter đang được phát triển",
        )

    return router
