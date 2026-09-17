"""Browser-safe VBS sessions, current-task lookup, and direct DRES submission.

This router accepts one participant ID and one typed answer. It never reads or
writes answer workspaces; the opaque task scope and DRES session remain private
to the backend service.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from hcmai.api.contracts.vbs import (
    VbsDirectSubmissionNotRecorded,
    VbsDirectSubmissionOutcome,
    VbsDirectSubmissionRecorded,
    VbsDirectSubmissionRequest,
    VbsDirectSubmissionUnknown,
    VbsEvaluationSummary,
    VbsSessionConnectRequest,
    VbsSessionStatus,
    VbsTaskResponse,
    VbsTaskTemplateItem,
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
    """Expose private participant sessions and stateless one-answer routes."""

    router = APIRouter()

    def _dres() -> Any:
        service = service_container.get("vbs_service")
        if service is None:
            _raise_api_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "DRES_NOT_CONFIGURED",
                "DRES session service is not configured",
            )
        return service

    @router.post("/api/v1/vbs/session/connect", response_model=VbsSessionStatus)
    async def connect_session(data: VbsSessionConnectRequest) -> VbsSessionStatus:
        """Connect the backend credential mapped to the supplied participant."""

        service = _dres()
        try:
            return VbsSessionStatus.model_validate(await service.connect(data.user_id))
        except DresUnknownUserError:
            _raise_api_error(
                status.HTTP_403_FORBIDDEN,
                "USER_NOT_CONFIGURED",
                "VBS user ID has no backend DRES credential mapping",
            )
        except DresAuthenticationError:
            _raise_api_error(
                status.HTTP_401_UNAUTHORIZED,
                "DRES_LOGIN_REJECTED",
                "DRES rejected the configured participant credentials",
            )
        except DresError as error:
            _raise_api_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "DRES_UNAVAILABLE",
                "DRES session could not be established",
            )

    @router.get("/api/v1/vbs/session/{user_id}", response_model=VbsSessionStatus)
    async def get_session_status(user_id: str) -> VbsSessionStatus:
        """Return one participant's connection state without its session token."""

        return VbsSessionStatus.model_validate(_dres().session_status(user_id))

    @router.delete("/api/v1/vbs/session/{user_id}", response_model=VbsSessionStatus)
    async def disconnect_session(user_id: str) -> VbsSessionStatus:
        """Evict only the selected participant's in-memory DRES session."""

        try:
            return VbsSessionStatus.model_validate(await _dres().disconnect(user_id))
        except DresError:
            _raise_api_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "DRES_UNAVAILABLE",
                "DRES session could not be cleared",
            )

    @router.get("/api/v1/vbs/evaluations/{user_id}", response_model=list[VbsEvaluationSummary])
    async def list_evaluations(user_id: str) -> list[VbsEvaluationSummary]:
        """List DRES evaluations and their task templates for a connected participant."""

        service = _dres()
        _require_connected(service, user_id)
        try:
            evaluations = await service.list_evaluations(user_id)
            summaries = []
            for ev in evaluations:
                summaries.append(
                    VbsEvaluationSummary(
                        id=ev.id,
                        name=ev.name,
                        type=ev.type,
                        status=ev.status,
                        template_id=ev.template_id,
                        task_templates=[
                            VbsTaskTemplateItem(
                                name=t.name,
                                task_group=t.task_group,
                                task_type=t.task_type,
                                duration=t.duration,
                            )
                            for t in ev.task_templates
                        ],
                    )
                )
            return summaries
        except DresError:
            _raise_api_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "DRES_EVALUATION_LIST_UNAVAILABLE",
                "Could not list evaluations from DRES",
            )

    @router.get("/api/v1/vbs/task/{user_id}", response_model=VbsTaskResponse)
    async def current_task(
        user_id: str,
        evaluation_id: str | None = None,
        task_name: str | None = None,
    ) -> VbsTaskResponse:
        """Return safe active or selected task metadata for a connected participant."""

        service = _dres()
        _require_connected(service, user_id)
        scope = await _resolve_scope(
            service,
            user_id,
            evaluation_id=evaluation_id,
            task_name=task_name,
        )
        return VbsTaskResponse(
            user_id=user_id,
            evaluation_id=scope.evaluation_id,
            task_scope_key=scope.task_scope_key,
            task_name=scope.task_name,
            task_group=scope.task_group,
            task_type=scope.task_type,
            duration=scope.duration,
        )

    @router.post(
        "/api/v1/vbs/submit",
        response_model=VbsDirectSubmissionOutcome,
    )
    async def submit_one(data: VbsDirectSubmissionRequest) -> VbsDirectSubmissionOutcome:
        """Validate the live task and forward exactly one answer once."""

        service = _dres()
        _require_connected(service, data.user_id)
        scope = await _resolve_scope(
            service,
            data.user_id,
            evaluation_id=data.evaluation_id,
            task_name=data.task_name,
        )

        if data.expected_task_scope_key != scope.task_scope_key:
            _raise_api_error(
                status.HTTP_409_CONFLICT,
                "TASK_SCOPE_MISMATCH",
                "The active DRES task changed; reopen the submission popup",
            )

        task_group = scope.task_group.strip().upper()
        task_type = scope.task_type.strip().upper()

        is_qa = task_type == "VQA" or task_group in {"VQA", "QA"} or "QUESTION ANSWERING" in task_type
        is_temporal = (
            task_type in {"KIS", "AVS", "TRAKE"}
            or task_group in {"KIS", "AVS", "TRAKE"}
            or any(k in task_type for k in ("KNOWN ITEM", "AD-HOC", "ADHOC", "VIDEO SEARCH"))
        )

        if not is_qa and not is_temporal:
            _raise_api_error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "UNSUPPORTED_DRES_TASK_TYPE",
                "The active DRES task type is not supported for direct submission",
            )
        expected_kind = "TEXT" if is_qa else "TEMPORAL"
        if data.answer.kind != expected_kind:
            _raise_api_error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "ANSWER_KIND_MISMATCH",
                f"The active {task_type} task requires a {expected_kind.lower()} answer",
            )

        try:
            if data.answer.kind == "TEMPORAL":
                mapped_answer = service.temporal_range_answer(
                    data.answer.video_id,
                    data.answer.start_ms,
                    data.answer.end_ms,
                )
            else:
                mapped_answer = ApiClientAnswer(text=data.answer.text)
        except ValueError as error:
            _raise_api_error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "INVALID_ANSWER",
                str(error),
            )

        payload = ApiClientSubmission(
            answer_sets=[ApiClientAnswerSet(
                task_name=scope.task_name,
                answers=[mapped_answer],
            )],
        )

        try:
            dres_status = await service.submit(
                data.user_id,
                scope.evaluation_id,
                payload,
                task_name=scope.task_name,
            )
        except DresAuthenticationError:
            return VbsDirectSubmissionNotRecorded(
                state="NOT_RECORDED",
                recorded=False,
                verdict=None,
                reason="DRES_AUTH_REJECTED",
                message="DRES rejected this participant session; reconnect before submitting again",
            )
        except DresRejectedSubmissionError as error:
            err_msg = str(error)
            prefix = "DRES rejected submit answers: "
            clean_msg = err_msg[len(prefix):] if err_msg.startswith(prefix) else err_msg
            return VbsDirectSubmissionNotRecorded(
                state="NOT_RECORDED",
                recorded=False,
                verdict=None,
                reason="DRES_REJECTED",
                message=clean_msg or "DRES rejected this answer; review it before submitting again",
            )
        except (DresUnavailableError, DresError) as error:
            return VbsDirectSubmissionUnknown(
                state="UNKNOWN",
                recorded=None,
                verdict=None,
                message=f"DRES outcome is unknown ({error}); check DRES before manually submitting again",
            )

        return VbsDirectSubmissionRecorded(
            state="RECORDED",
            recorded=True,
            verdict=dres_status.submission,
            message="DRES recorded the answer",
        )

    @router.post("/api/v1/vbs/task/activate")
    async def activate_task(
        evaluation_id: str,
        task_name: str,
    ) -> dict[str, Any]:
        """Activate and start a specific task run if admin credentials are configured."""

        service = _dres()
        if not service.settings.admin_credential:
            _raise_api_error(
                status.HTTP_403_FORBIDDEN,
                "ADMIN_NOT_CONFIGURED",
                "Admin credentials are not configured on the backend",
            )
        success = await service.ensure_task_running(evaluation_id, task_name)
        if not success:
            _raise_api_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "TASK_ACTIVATION_FAILED",
                f"Failed to activate task '{task_name}' on DRES",
            )
        return {"status": "ok", "task_name": task_name, "evaluation_id": evaluation_id}

    return router


async def _resolve_scope(
    service: Any,
    user_id: str,
    *,
    evaluation_id: str | None = None,
    task_name: str | None = None,
) -> Any:
    """Resolve current or selected task metadata while keeping failures browser-safe."""

    try:
        return await service.resolve_scope(
            user_id,
            evaluation_id=evaluation_id,
            task_name=task_name,
        )
    except DresNoActiveTaskError:
        _raise_api_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "NO_ACTIVE_DRES_TASK",
            "DRES has no usable current task",
        )
    except DresError:
        _raise_api_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "DRES_TASK_UNAVAILABLE",
            "Unable to resolve the active DRES task",
        )
    except DresNoActiveTaskError:
        _raise_api_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "NO_ACTIVE_DRES_TASK",
            "DRES has no usable current task",
        )
    except DresError:
        _raise_api_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "DRES_TASK_UNAVAILABLE",
            "Unable to resolve the active DRES task",
        )


def _require_connected(service: Any, user_id: str) -> None:
    """Reject disconnected IDs without triggering an implicit DRES login."""

    if not service.session_status(user_id).get("connected"):
        _raise_api_error(
            status.HTTP_401_UNAUTHORIZED,
            "PARTICIPANT_NOT_CONNECTED",
            "Connect this VBS participant before looking up tasks or submitting answers",
        )


def _raise_api_error(status_code: int, code: str, message: str) -> None:
    """Raise a FastAPI error with a stable browser-facing code and message."""

    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


__all__ = ["create_vbs_router"]
