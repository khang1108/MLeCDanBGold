"""Thin HTTP transport for in-process temporal exploration branches.

This module owns bounded handle lifecycle and HTTP error mapping. It delegates
all scoring, constraint updates, and path decoding to ``TemporalExploration``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Annotated, Any, Iterator
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts.exploration import (
    ExplorationActionRequest,
    ExplorationConditionsResponse,
    ExplorationEnvelope,
    ExplorationOpenRequest,
    ExplorationPathResponse,
    ExplorationViewResponse,
)
from hcmai.orchestration.workflows.temporal_exploration import (
    ExplorationConflict,
    ExplorationUnavailable,
    ExplorationView,
    QueryBinding,
    TemporalExploration,
)

_MAX_HANDLES = 16
_HANDLE_TTL = timedelta(minutes=30)


@dataclass(slots=True)
class _RegistryEntry:
    """Retain one branch plus lifecycle bookkeeping for a public handle."""

    branch: TemporalExploration | None
    scoring_revision: str
    last_access: datetime
    active_operations: int = 0
    closing: bool = False


class ExplorationRegistry:
    """Bound one worker's local exploration branches and their handles."""

    def __init__(self, *, max_handles: int = _MAX_HANDLES) -> None:
        """Create an empty registry with one startup-owned scoring generation."""

        self._max_handles = max_handles
        self._scoring_revision = str(uuid4())
        self._entries: dict[str, _RegistryEntry] = {}
        self._lock = Lock()

    @property
    def scoring_revision(self) -> str:
        """Return the immutable generation issued for this app lifetime."""

        return self._scoring_revision

    def reserve(self) -> str | None:
        """Reserve capacity for an open before doing blocking scoring work."""

        with self._lock:
            self._expire_idle_locked()
            if len(self._entries) >= self._max_handles:
                return None
            handle = str(uuid4())
            self._entries[handle] = _RegistryEntry(
                branch=None,
                scoring_revision=self._scoring_revision,
                last_access=datetime.now(UTC),
            )
            return handle

    def publish(self, handle: str, branch: TemporalExploration) -> bool:
        """Publish a completed open only while its reservation still exists."""

        with self._lock:
            entry = self._entries.get(handle)
            if entry is None:
                return False
            entry.branch = branch
            entry.last_access = datetime.now(UTC)
            return True

    def discard(self, handle: str) -> None:
        """Drop an unused reservation or a cancelled open result."""

        with self._lock:
            self._entries.pop(handle, None)

    @contextmanager
    def borrow(self, handle: UUID) -> Iterator[_RegistryEntry]:
        """Borrow a branch without holding registry locking during core work."""

        key = str(handle)
        with self._lock:
            self._expire_idle_locked()
            entry = self._entries.get(key)
            if entry is None or entry.branch is None or entry.closing:
                raise KeyError(key)
            entry.active_operations += 1
            entry.last_access = datetime.now(UTC)
        try:
            yield entry
        finally:
            with self._lock:
                entry.active_operations -= 1
                entry.last_access = datetime.now(UTC)

    def close(self, handle: UUID, expected_revision: int) -> None:
        """Close and remove one idle branch as one handle lifecycle operation."""

        key = str(handle)
        with self._lock:
            self._expire_idle_locked()
            entry = self._entries.get(key)
            if entry is None or entry.branch is None:
                raise KeyError(key)
            if entry.active_operations or entry.closing:
                raise ExplorationConflict("exploration handle is busy")

            # Mark closing before releasing the registry lock so another
            # request cannot borrow a branch after its core has been closed.
            entry.active_operations += 1
            entry.closing = True
            branch = entry.branch

        try:
            branch.close(expected_revision=expected_revision)
        except BaseException:
            with self._lock:
                if self._entries.get(key) is entry:
                    entry.active_operations -= 1
                    entry.closing = False
                    entry.last_access = datetime.now(UTC)
            raise

        with self._lock:
            # Removal happens in this same operation, after a successful core
            # close, so an unreachable closed branch cannot remain registered.
            if self._entries.get(key) is entry:
                del self._entries[key]

    def _expire_idle_locked(self) -> None:
        """Lazily remove inactive, idle branches while retaining active work.

        The lock protects reservation/removal bookkeeping only; scoring always
        happens after release so one slow decode cannot block other handles.
        """

        cutoff = datetime.now(UTC) - _HANDLE_TTL
        expired = [
            handle
            for handle, entry in self._entries.items()
            if entry.branch is not None
            and entry.active_operations == 0
            and entry.last_access < cutoff
        ]
        for handle in expired:
            del self._entries[handle]


def create_exploration_router(service_container: dict[str, Any]) -> APIRouter:
    """Create exploration routes over the immutable startup search service."""

    registry = service_container.get("exploration_registry")
    if not isinstance(registry, ExplorationRegistry):
        registry = ExplorationRegistry()
        service_container["exploration_registry"] = registry

    router = APIRouter()

    @router.post("/api/v1/exploration", response_model=ExplorationEnvelope)
    async def open_exploration(request: ExplorationOpenRequest) -> ExplorationEnvelope:
        """Open a bounded local branch from the supplied retrieval snapshot."""

        temporal = _temporal_service(service_container.get("service"))
        if temporal is None:
            raise _unavailable("Temporal exploration service not initialized")
        handle = registry.reserve()
        if handle is None:
            raise _unavailable("Temporal exploration capacity is full")
        try:
            branch, view = await run_in_threadpool(
                _open_branch,
                temporal,
                request,
                registry.scoring_revision,
            )
            if not registry.publish(handle, branch):
                # A cancelled caller removed the reservation while scoring ran.
                return _missing_handle(handle)
        except ExplorationUnavailable as error:
            registry.discard(handle)
            raise _unavailable(str(error)) from error
        except ValueError as error:
            registry.discard(handle)
            raise _invalid_input(str(error)) from error
        except BaseException:
            # Cancellation must not leave a reserved handle consuming capacity.
            registry.discard(handle)
            raise
        return _envelope(handle, registry.scoring_revision, view)

    @router.get("/api/v1/exploration/{handle}", response_model=ExplorationEnvelope)
    async def get_exploration(handle: UUID) -> ExplorationEnvelope:
        """Return the latest immutable snapshot for reconciliation after a conflict."""

        try:
            with registry.borrow(handle) as entry:
                view = await run_in_threadpool(_current, entry.branch)
                return _envelope(str(handle), entry.scoring_revision, view)
        except KeyError as error:
            raise _missing_handle(str(handle)) from error
        except ExplorationUnavailable as error:
            raise _unavailable(str(error)) from error

    @router.post(
        "/api/v1/exploration/{handle}/actions",
        response_model=ExplorationEnvelope,
    )
    async def act_on_exploration(
        handle: UUID,
        request: ExplorationActionRequest,
    ) -> ExplorationEnvelope:
        """Apply one guarded feedback action without retrying a mutation."""

        try:
            with registry.borrow(handle) as entry:
                view = await run_in_threadpool(_apply_action, entry.branch, request)
                return _envelope(str(handle), entry.scoring_revision, view)
        except KeyError as error:
            raise _missing_handle(str(handle)) from error
        except ExplorationConflict as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error
        except ExplorationUnavailable as error:
            raise _unavailable(str(error)) from error
        except ValueError as error:
            raise _invalid_input(str(error)) from error

    @router.delete(
        "/api/v1/exploration/{handle}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def close_exploration(
        handle: UUID,
        expected_revision: Annotated[int, Query(ge=1)],
    ) -> Response:
        """Close an idle branch after its revision guard succeeds."""

        try:
            await run_in_threadpool(registry.close, handle, expected_revision)
        except KeyError as error:
            raise _missing_handle(str(handle)) from error
        except ExplorationConflict as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error
        except ExplorationUnavailable as error:
            raise _unavailable(str(error)) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router


def _temporal_service(service: object) -> object | None:
    """Read the existing shared temporal service without creating another core."""

    return getattr(getattr(service, "kis", None), "temporal", None)


def _open_branch(
    temporal: object,
    request: ExplorationOpenRequest,
    scoring_revision: str,
) -> tuple[TemporalExploration, ExplorationView]:
    """Open one real branch in the worker thread used for scoring and decoding."""

    binding = QueryBinding(
        retrieval_plan=request.seed.to_plan(),
        semantic_revision=request.seed.semantic_revision,
        event_version=str(uuid4()),
        use_dense=request.seed.use_dense,
        use_bm25=request.seed.use_bm25,
        scoring_revision=scoring_revision,
    )
    branch = TemporalExploration(temporal)  # type: ignore[arg-type]
    return branch, branch.open(binding, request.video_id, request.window)


def _current(branch: TemporalExploration | None) -> ExplorationView:
    """Read a borrowed branch snapshot in the worker thread."""

    if branch is None:
        raise ExplorationUnavailable("temporal exploration is not open")
    return branch.current()


def _apply_action(
    branch: TemporalExploration | None,
    request: ExplorationActionRequest,
) -> ExplorationView:
    """Delegate an allowed action to the real branch lifecycle."""

    if branch is None:
        raise ExplorationUnavailable("temporal exploration is not open")
    if request.action == "undo":
        return branch.undo(
            expected_revision=request.expected_revision,
            event_version=str(request.event_version),
            scoring_revision=str(request.scoring_revision),
        )
    return branch.apply(
        expected_revision=request.expected_revision,
        event_version=str(request.event_version),
        scoring_revision=str(request.scoring_revision),
        action=request.action,
        event_index=request.event_index,
        interval=request.interval,
    )


def _envelope(
    handle: str,
    scoring_revision: str,
    view: ExplorationView,
) -> ExplorationEnvelope:
    """Project the core's immutable tuples into JSON array transport fields."""

    return ExplorationEnvelope(
        handle=handle,
        scoring_revision=scoring_revision,
        view=ExplorationViewResponse(
            revision=view.revision,
            event_version=view.event_version,
            video_id=view.video_id,
            conditions=ExplorationConditionsResponse(
                window=view.conditions.window,
                confirmed=list(view.conditions.confirmed),
                rejected=[list(row) for row in view.conditions.rejected],
            ),
            status=view.status,
            paths=[
                ExplorationPathResponse(
                    video_id=path.video_id,
                    score=path.score,
                    frame_ids=list(path.frame_ids),
                    frame_idxs=list(path.frame_idxs),
                    timestamps_ms=list(path.timestamps_ms),
                )
                for path in view.paths
            ],
            changed_event_indices=list(view.changed_event_indices),
            comparison_available=view.comparison_available,
            can_undo=view.can_undo,
        ),
    )


def _missing_handle(handle: str) -> HTTPException:
    """Build the consistent missing-handle response."""

    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"exploration handle {handle} was not found",
    )


def _unavailable(detail: str) -> HTTPException:
    """Build the consistent unavailable-dependency response."""

    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


def _invalid_input(detail: str) -> HTTPException:
    """Build the consistent semantic request-validation response."""

    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=detail,
    )
