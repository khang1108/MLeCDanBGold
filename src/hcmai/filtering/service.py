"""Execute exact, deterministic metadata filtering over a published catalog.

The service owns predicate construction, counts, canonical ordering, and page
materialization. It does not perform retrieval or expose SQLite/source paths.
"""

from __future__ import annotations

import math
import sqlite3

from dataclasses import dataclass
from typing import Any

from hcmai.api.contracts.filter import FilterRequest, FilterResponse, FilterResult
from hcmai.common.utils.logging import get_logger

from .catalog import FilterCatalog, FilterCatalogUnavailableError


logger = get_logger(__name__)


class FilterServiceUnavailableError(RuntimeError):
    """Raised when Filter serving cannot borrow its optional catalog."""


@dataclass(frozen=True)
class _Predicate:
    """One parameterized WHERE clause shared by count and page queries."""

    sql: str
    parameters: tuple[Any, ...]


class FilterService:
    """Serve exact metadata pages from one bounded read-only catalog."""

    def __init__(self, catalog: FilterCatalog) -> None:
        """Bind the service to one already-validated catalog."""

        self.catalog = catalog

    def filter(self, request: FilterRequest) -> FilterResponse:
        """Return one canonical page matching all active exact predicates."""

        predicate = self._predicate(request)
        try:
            with self.catalog.connection() as connection:
                total_results = connection.execute(
                    f"SELECT COUNT(*) AS count FROM frames {predicate.sql}",
                    predicate.parameters,
                ).fetchone()["count"]
                total_pages = (
                    math.ceil(total_results / request.frames_per_pages)
                    if total_results
                    else 0
                )
                rows = connection.execute(
                    f"""
                    SELECT frame_id, video_id, frame_idx, timestamp_ms, folder_id,
                           title, caption, ocr, asr
                    FROM frames
                    {predicate.sql}
                    ORDER BY video_id, timestamp_ms, frame_idx, frame_id
                    LIMIT ? OFFSET ?
                    """,
                    (
                        *predicate.parameters,
                        request.frames_per_pages,
                        (request.page_id - 1) * request.frames_per_pages,
                    ),
                ).fetchall()
                objects = _page_objects(connection, [row["frame_id"] for row in rows])
        except FilterCatalogUnavailableError as error:
            raise FilterServiceUnavailableError(str(error)) from error

        results = [
            FilterResult(
                frame_id=row["frame_id"],
                video_id=row["video_id"],
                frame_idx=row["frame_idx"],
                timestamp_ms=row["timestamp_ms"],
                folder_id=row["folder_id"],
                title=row["title"],
                caption=row["caption"],
                ocr=row["ocr"],
                objects=objects.get(row["frame_id"], {}),
                asr=row["asr"],
            )
            for row in rows
        ]
        return FilterResponse(
            page_id=request.page_id,
            frames_per_pages=request.frames_per_pages,
            total_results=total_results,
            total_pages=total_pages,
            results=results,
        )

    def health(self) -> dict[str, object]:
        """Return safe capability facts without source or database paths."""

        return {
            "ready": True,
            "catalog_version": self.catalog.info.catalog_version,
            "frame_count": self.catalog.info.frame_count,
        }

    def close(self) -> None:
        """Close the owned catalog pool during application shutdown."""

        self.catalog.close()

    def _predicate(self, request: FilterRequest) -> _Predicate:
        """Build one deterministic parameterized AND predicate."""

        clauses: list[str] = []
        parameters: list[Any] = []
        if request.folder_id is not None:
            clauses.append("frames.folder_id = ?")
            parameters.append(request.folder_id)
        if request.video_id is not None:
            clauses.append("frames.video_id = ?")
            parameters.append(request.video_id)

        filters = request.metadata_filters
        availability = self.catalog.info.availability
        text_fields = (
            ("title", filters.title, availability.title),
            ("asr", filters.asr, availability.asr),
            ("caption", filters.caption, availability.caption),
            ("ocr", filters.ocr, availability.ocr),
        )
        for field, value, available in text_fields:
            if value is None:
                continue
            if not available:
                logger.warning(
                    "Filter predicate ignored because modality is unavailable: %s",
                    field,
                )
                continue
            clauses.append(f"instr(frames.{field}_norm, ?) > 0")
            parameters.append(value)

        if filters.objects and not availability.objects:
            logger.warning(
                "Filter predicate ignored because modality is unavailable: objects"
            )
        elif availability.objects:
            for label, count in sorted(filters.objects.items()):
                clauses.append(
                    """
                    EXISTS (
                        SELECT 1 FROM frame_objects AS fo
                        WHERE fo.frame_id = frames.frame_id
                          AND fo.label_norm = ?
                          AND fo.object_count = ?
                    )
                    """.strip()
                )
                parameters.extend((label, count))

        sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return _Predicate(sql=sql, parameters=tuple(parameters))


def _page_objects(
    connection: sqlite3.Connection,
    frame_ids: list[str],
) -> dict[str, dict[str, int]]:
    """Fetch exact object multiplicity for only the materialized result page."""

    if not frame_ids:
        return {}
    placeholders = ", ".join("?" for _ in frame_ids)
    rows = connection.execute(
        f"""
        SELECT frame_id, label_norm, object_count
        FROM frame_objects
        WHERE frame_id IN ({placeholders})
        ORDER BY frame_id, label_norm
        """,
        frame_ids,
    ).fetchall()
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        result.setdefault(row["frame_id"], {})[row["label_norm"]] = row[
            "object_count"
        ]
    return result

