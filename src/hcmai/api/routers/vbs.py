"""Browser-safe VBS session and durable DRES submission routes.

This transport accepts participant IDs and answer revisions only. It resolves
all task state and builds every DRES payload on the backend before forwarding.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from hcmai.api.contracts.vbs import (
    VbsAvsSubmissionRequest,
    VbsSessionConnectRequest,
    VbsSessionStatus,
    VbsSingleSubmissionRequest,
    VbsSubmissionResolutionRequest,
    VbsSubmissionResponse,
)
from hcmai.api.contracts.workspace import AnswerWorkspaceEvent, AnswerWorkspaceSnapshot
from hcmai.api.history import (
    AnswerSubmissionInFlight,
    AnswerWorkspaceConflict,
    AnswerWorkspaceTaskScopeMismatch,
    WorkspaceStore,
)
from hcmai.vbs.client import (
    DresAuthenticationError,
    DresError,
    DresNoActiveTaskError,
    DresRejectedSubmissionError,
    DresUnavailableError,
)
from hcmai.vbs.models import ApiClientAnswer, ApiClientAnswerSet, ApiClientSubmission
from hcmai.vbs.service import DresUnknownUserError


def create_vbs_router(service_container: dict[str, Any]) -> APIRouter:
    """Expose participant session, KIS/VQA/AVS, and attempt-resolution routes."""

    router = APIRouter()

    def _dres() -> Any:
        service = service_container.get("vbs_service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DRES session service is not configured",
            )
        return service

    def _store() -> WorkspaceStore:
        store = service_container.get("workspace_store")
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Answer workspace database is not configured",
            )
        return store

    @router.post("/api/v1/vbs/session/connect", response_model=VbsSessionStatus)
    async def connect_session(data: VbsSessionConnectRequest) -> VbsSessionStatus:
        """Connect a backend-mapped DRES identity using only the VBS user ID."""

        service = _dres()
        try:
            return VbsSessionStatus.model_validate(await service.connect(data.user_id))
        except DresUnknownUserError as error:
            raise HTTPException(status_code=403, detail="VBS user ID has no backend DRES credential mapping") from error
        except DresAuthenticationError as error:
            raise HTTPException(status_code=401, detail="DRES rejected the configured participant credentials") from error
        except DresError as error:
            raise HTTPException(status_code=503, detail="DRES session could not be established") from error

    @router.get("/api/v1/vbs/session/{user_id}", response_model=VbsSessionStatus)
    async def get_session_status(user_id: str) -> VbsSessionStatus:
        """Return whether one participant has a private cached DRES session."""

        return VbsSessionStatus.model_validate(_dres().session_status(user_id))

    @router.delete("/api/v1/vbs/session/{user_id}", response_model=VbsSessionStatus)
    async def disconnect_session(user_id: str) -> VbsSessionStatus:
        """Remove only the selected participant's in-memory DRES session."""

        try:
            return VbsSessionStatus.model_validate(await _dres().disconnect(user_id))
        except DresError as error:
            raise HTTPException(status_code=503, detail="DRES session could not be cleared") from error

    @router.post("/api/v1/vbs/submit/kis", response_model=VbsSubmissionResponse)
    async def submit_kis(data: VbsSingleSubmissionRequest):
        """Forward one revision-checked FRAME answer as a KIS point."""

        return await _forward_submission(
            service_container,
            kind="KIS",
            user_id=data.user_id,
            requested_task_scope_key=data.task_scope_key,
            expected_workspace_revision=data.expected_workspace_revision,
            candidates=[{
                "candidate_id": data.candidate_id,
                "expected_revision": data.expected_revision,
            }],
        )

    @router.post("/api/v1/vbs/submit/vqa", response_model=VbsSubmissionResponse)
    async def submit_vqa(data: VbsSingleSubmissionRequest):
        """Forward one revision-checked TEXT answer as a VQA response."""

        return await _forward_submission(
            service_container,
            kind="VQA",
            user_id=data.user_id,
            requested_task_scope_key=data.task_scope_key,
            expected_workspace_revision=data.expected_workspace_revision,
            candidates=[{
                "candidate_id": data.candidate_id,
                "expected_revision": data.expected_revision,
            }],
        )

    @router.post("/api/v1/vbs/submit/avs", response_model=VbsSubmissionResponse)
    async def submit_avs(data: VbsAvsSubmissionRequest):
        """Forward every ordered eligible FRAME candidate in one DRES call."""

        return await _forward_submission(
            service_container,
            kind="AVS",
            user_id=data.user_id,
            requested_task_scope_key=data.task_scope_key,
            expected_workspace_revision=data.expected_workspace_revision,
            candidates=[candidate.model_dump() for candidate in data.candidates],
        )

    @router.post(
        "/api/v1/vbs/submission-attempts/{attempt_id}/resolve",
        response_model=VbsSubmissionResponse,
    )
    async def resolve_submission_attempt(
        attempt_id: str,
        data: VbsSubmissionResolutionRequest,
    ) -> VbsSubmissionResponse:
        """Apply an operator-confirmed outcome to one current UNKNOWN attempt."""

        service = _dres()
        if not service.session_status(data.user_id).get("connected"):
            raise HTTPException(status_code=401, detail="Connect the VBS participant before resolving an attempt")
        store = _store()
        accepted = data.outcome == "accepted"
        try:
            attempt = store.get_submission_attempt(attempt_id)
            workspace = store.resolve_unknown_submission(attempt_id, accepted=accepted)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Submission attempt was not found") from error
        except AnswerWorkspaceConflict as error:
            raise HTTPException(status_code=409, detail="Attempt is not the current UNKNOWN workspace reservation") from error
        await _broadcast(
            service_container,
            "answer.submitted" if accepted else "answer.error",
            workspace,
        )
        return VbsSubmissionResponse(
            attempt_id=attempt.attempt_id,
            state="ACCEPTED" if accepted else "NOT_ACCEPTED",
            accepted=accepted,
            message="Unknown DRES outcome was resolved from the operator decision",
            workspace=workspace,
        )

    return router


async def _forward_submission(
    container: dict[str, Any],
    *,
    kind: str,
    user_id: str,
    requested_task_scope_key: str,
    expected_workspace_revision: int,
    candidates: list[dict[str, object]],
) -> VbsSubmissionResponse | JSONResponse:
    """Resolve, reserve, and send exactly one frozen DRES answer set."""

    service = container.get("vbs_service")
    store = container.get("workspace_store")
    if service is None:
        raise HTTPException(status_code=503, detail="DRES session service is not configured")
    if store is None:
        raise HTTPException(status_code=503, detail="Answer workspace database is not configured")
    if not service.session_status(user_id).get("connected"):
        raise HTTPException(status_code=401, detail="Connect the VBS participant before submitting answers")
    try:
        scope = await service.resolve_scope(user_id)
    except DresNoActiveTaskError as error:
        raise HTTPException(status_code=503, detail="DRES has no usable current task") from error
    except DresError as error:
        raise HTTPException(status_code=503, detail="Unable to resolve the active DRES task") from error
    if requested_task_scope_key != scope.task_scope_key:
        raise HTTPException(status_code=409, detail="TASK_SCOPE_MISMATCH: the active DRES task changed")

    try:
        workspace = store.get_answer_workspace(
            scope.evaluation_id,
            scope.task_scope_key,
            scope.task_name,
        )
        if workspace.task_scope_mismatch:
            raise AnswerWorkspaceTaskScopeMismatch(
                workspace.evaluation_id,
                workspace.task_scope_key,
                scope.evaluation_id,
                scope.task_scope_key,
            )
        payload = _build_submission_payload(kind, scope.task_name, candidates, workspace, service)
        # Resolve again after reading candidates so a task transition during
        # local payload construction cannot cross the durable reservation.
        latest_scope = await service.resolve_scope(user_id)
        if (latest_scope.evaluation_id, latest_scope.task_scope_key) != (
            scope.evaluation_id,
            scope.task_scope_key,
        ):
            raise HTTPException(status_code=409, detail="TASK_SCOPE_MISMATCH: the active DRES task changed")
        attempt = store.reserve_submission(
            kind=kind,
            user_id=user_id,
            evaluation_id=scope.evaluation_id,
            task_scope_key=scope.task_scope_key,
            task_name=scope.task_name,
            expected_workspace_revision=expected_workspace_revision,
            candidates=candidates,
            dres_payload=payload,
        )
    except AnswerWorkspaceTaskScopeMismatch as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except AnswerSubmissionInFlight as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except AnswerWorkspaceConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    forwarding = store.get_answer_workspace(
        scope.evaluation_id,
        scope.task_scope_key,
        scope.task_name,
    )
    await _broadcast(container, "answer.submitting", forwarding)
    frozen = json.loads(attempt.snapshot_json)
    submission = ApiClientSubmission.model_validate(frozen["dres_payload"])

    try:
        send_scope = await service.resolve_scope(user_id)
    except DresError:
        return await _release_unsent_attempt(
            container,
            attempt.attempt_id,
            status_code=503,
            dres_status="task scope unavailable before send",
            message="Unable to verify the active DRES task; the answer was not sent",
        )
    if (send_scope.evaluation_id, send_scope.task_scope_key) != (
        scope.evaluation_id,
        scope.task_scope_key,
    ):
        return await _release_unsent_attempt(
            container,
            attempt.attempt_id,
            status_code=409,
            dres_status="task scope changed before send",
            message="The active DRES task changed before sending; review the workspace and retry",
        )

    try:
        dres_status = await service.submit(user_id, scope.evaluation_id, submission)
    except (DresRejectedSubmissionError, DresAuthenticationError):
        workspace = store.complete_submission(
            attempt.attempt_id,
            accepted=False,
            dres_status="rejected",
        )
        await _broadcast(container, "answer.error", workspace)
        return JSONResponse(
            status_code=502,
            content=VbsSubmissionResponse(
                attempt_id=attempt.attempt_id,
                state="NOT_ACCEPTED",
                accepted=False,
                message="DRES rejected the submission; review the answer before trying again",
                workspace=workspace,
            ).model_dump(mode="json"),
        )
    except (DresUnavailableError, DresError):
        workspace = store.mark_submission_unknown(attempt.attempt_id)
        await _broadcast(container, "answer.error", workspace)
        return JSONResponse(
            status_code=504,
            content=VbsSubmissionResponse(
                attempt_id=attempt.attempt_id,
                state="UNKNOWN",
                accepted=None,
                message="DRES outcome is unknown; verify DRES before resolving this attempt",
                workspace=workspace,
            ).model_dump(mode="json"),
        )

    workspace = store.complete_submission(
        attempt.attempt_id,
        accepted=True,
        dres_status="accepted",
    )
    await _broadcast(container, "answer.submitted", workspace)
    return VbsSubmissionResponse(
        attempt_id=attempt.attempt_id,
        state="ACCEPTED",
        accepted=True,
        verdict=dres_status.submission,
        message="DRES accepted the submission",
        workspace=workspace,
    )


async def _release_unsent_attempt(
    container: dict[str, Any],
    attempt_id: str,
    *,
    status_code: int,
    dres_status: str,
    message: str,
) -> JSONResponse:
    """Finalize a still-unsent reservation without marking its answer accepted."""

    store = container["workspace_store"]
    workspace = store.complete_submission(
        attempt_id,
        accepted=False,
        dres_status=dres_status,
    )
    await _broadcast(container, "answer.error", workspace)
    return JSONResponse(
        status_code=status_code,
        content=VbsSubmissionResponse(
            attempt_id=attempt_id,
            state="NOT_ACCEPTED",
            accepted=False,
            message=message,
            workspace=workspace,
        ).model_dump(mode="json"),
    )


def _build_submission_payload(
    kind: str,
    task_name: str,
    candidates: list[dict[str, object]],
    workspace: AnswerWorkspaceSnapshot,
    service: Any,
) -> dict[str, object]:
    """Map reviewed candidate values into one exact DRES answerSet object."""

    by_id = {candidate.candidate_id: candidate for candidate in workspace.candidates}
    answers: list[ApiClientAnswer] = []
    for reference in candidates:
        candidate_id = reference["candidate_id"]
        candidate = by_id.get(candidate_id)
        if candidate is None or candidate.revision != reference["expected_revision"]:
            raise AnswerWorkspaceConflict("Candidate is missing or its revision changed")
        if kind == "VQA":
            if candidate.kind != "TEXT" or candidate.text is None:
                raise AnswerWorkspaceConflict("VQA requires one TEXT candidate")
            answers.append(ApiClientAnswer(text=candidate.text))
        else:
            if candidate.kind != "FRAME" or candidate.video_id is None or candidate.timestamp_ms is None:
                raise AnswerWorkspaceConflict("KIS/AVS require FRAME candidates")
            answers.append(service.temporal_answer(candidate.video_id, candidate.timestamp_ms))
    payload = ApiClientSubmission(
        answer_sets=[ApiClientAnswerSet(task_name=task_name, answers=answers)],
    )
    return payload.model_dump(by_alias=True, exclude_none=True)


async def _broadcast(
    container: dict[str, Any],
    event_type: str,
    workspace: AnswerWorkspaceSnapshot,
) -> None:
    """Notify workspace collaborators when a durable submission state changes."""

    callback = container.get("broadcast_workspace")
    if callback is not None:
        await callback(AnswerWorkspaceEvent(type=event_type, workspace=workspace))


__all__ = ["create_vbs_router"]
