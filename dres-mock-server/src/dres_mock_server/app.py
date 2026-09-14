"""HTTP surface for a standalone DRES v2 contract simulator.

This module owns DRES-compatible transport and local test controls. It does not
import HCMAI or make scoring decisions; all sessions and captured requests live
in an app-scoped in-memory state object.
"""

from __future__ import annotations

import asyncio
from importlib.resources import files
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response

from dres_mock_server.models import (
    ApiClientEvaluationInfo,
    ApiClientSubmission,
    ApiClientTaskTemplateInfo,
    ApiUser,
    ErrorStatus,
    LoginRequest,
    QueryResultLog,
    ResultLogScenarioRequest,
    SuccessStatus,
    SubmissionScenarioRequest,
    SuccessfulSubmissionsStatus,
    TaskControlRequest,
    DresVerdict,
)
from dres_mock_server.settings import MockSettings
from dres_mock_server.state import CapturedRequest, MockState, ResponseScenario


_STATIC_ASSETS = {
    "monitor.css": ("text/css; charset=utf-8", "monitor.css"),
    "monitor.js": ("text/javascript; charset=utf-8", "monitor.js"),
}


def create_app(settings: MockSettings, state: MockState | None = None) -> FastAPI:
    """Create a new DRES mock app with settings and isolated runtime state."""

    app = FastAPI(
        title="Standalone DRES v2 Mock",
        description=(
            "A local DRES Client API contract simulator. It records requests in "
            "memory and does not score answer meaning."
        ),
        version="0.1.0",
        openapi_tags=[
            {
                "name": "Test Control",
                "description": (
                    "TEST CONTROL — LOCAL ONLY. These unauthenticated controls "
                    "must not be exposed publicly."
                ),
            }
        ],
    )
    mock_state = state or MockState(settings)
    app.state.mock_state = mock_state
    app.state.mock_settings = settings

    @app.middleware("http")
    async def mark_mock_responses(request: Request, call_next: Any):
        """Mark every HTTP response so clients can identify the local simulator."""

        response = await call_next(request)
        response.headers["X-DRES-Mock"] = "true"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Translate FastAPI validation errors without reflecting submitted values."""

        locations: list[str] = []
        for error in exc.errors():
            parts = error.get("loc", ())
            location = ".".join(str(part) for part in parts if isinstance(part, (str, int)))
            if location and location not in locations:
                locations.append(location)

        description = "request validation failed"
        if locations:
            description = f"{description}: {', '.join(locations[:4])}"
        return _error_response(400, description)

    @app.exception_handler(HTTPException)
    async def dres_http_error(_: Request, exc: HTTPException) -> JSONResponse:
        """Return DRES-shaped error bodies for endpoint and routing failures."""

        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        return _error_response(exc.status_code, detail)

    @app.post(
        "/api/v2/login",
        response_model=ApiUser,
        response_model_exclude_none=True,
        tags=["User"],
    )
    async def login(credentials: LoginRequest) -> ApiUser:
        """Validate one local account and return a DRES-compatible session."""

        session_id = mock_state.login(credentials.username, credentials.password)
        if session_id is None:
            raise HTTPException(status_code=401, detail="invalid credentials")
        return ApiUser(
            id=credentials.username,
            username=credentials.username,
            role="PARTICIPANT",
            session_id=session_id,
        )

    @app.get(
        "/api/v2/logout",
        response_model=SuccessStatus,
        tags=["User"],
    )
    async def logout(session: str | None = Query(default=None)) -> SuccessStatus:
        """Invalidate only the session token named by the DRES client."""

        if not session or not mock_state.logout(session):
            raise HTTPException(status_code=400, detail="invalid session")
        return SuccessStatus(status=True, description="session closed")

    @app.get(
        "/api/v2/client/evaluation/list",
        response_model=list[ApiClientEvaluationInfo],
        response_model_exclude_none=True,
        tags=["Evaluation Client"],
    )
    async def evaluation_list(session: str | None = Query(default=None)) -> list[ApiClientEvaluationInfo]:
        """List the one active synchronous evaluation after session validation."""

        _require_session(mock_state, session)
        current_task = mock_state.active_task()
        tasks = [current_task] if current_task is not None else []
        return [
            ApiClientEvaluationInfo(
                id=settings.evaluation_id,
                name=settings.evaluation_name,
                type=settings.evaluation_type,
                status="ACTIVE",
                template_id="template-vbs-local",
                teams=[],
                task_templates=tasks,
            )
        ]

    @app.get(
        "/api/v2/client/evaluation/currentTask/{evaluationId}",
        response_model=ApiClientTaskTemplateInfo,
        response_model_exclude_none=True,
        tags=["Evaluation Client"],
    )
    async def current_task(
        evaluationId: str,
        session: str | None = Query(default=None),
    ) -> ApiClientTaskTemplateInfo:
        """Return active template metadata without inventing a DRES task ID."""

        _require_session(mock_state, session)
        if evaluationId != settings.evaluation_id:
            raise HTTPException(status_code=404, detail="evaluation not found")
        task = mock_state.active_task()
        if task is None:
            raise HTTPException(status_code=404, detail="no active task")
        return task

    @app.post(
        "/api/v2/submit/{evaluationId}",
        tags=["Submission"],
    )
    async def submit(
        submission: ApiClientSubmission,
        evaluationId: str,
        session: str | None = Query(default=None),
    ) -> JSONResponse:
        """Capture one valid submission and return the configured mock outcome."""

        _require_session(mock_state, session)
        if evaluationId != settings.evaluation_id:
            raise HTTPException(status_code=404, detail="evaluation not found")

        record = mock_state.capture_submission(session, evaluationId, submission)
        if record is None:
            # Re-check under the state lock so a concurrent logout cannot turn
            # an unauthenticated request into a captured participant record.
            raise HTTPException(status_code=401, detail="invalid session")
        return await _captured_response(record, submission=True)

    @app.post(
        "/api/v2/log/result/{evaluationId}",
        tags=["Log"],
    )
    async def log_result(
        result_log: QueryResultLog,
        evaluationId: str,
        session: str | None = Query(default=None),
    ) -> JSONResponse:
        """Capture one result log without evaluating its retrieval quality."""

        _require_session(mock_state, session)
        if evaluationId != settings.evaluation_id:
            raise HTTPException(status_code=404, detail="evaluation not found")

        record = mock_state.capture_result_log(session, evaluationId, result_log)
        if record is None:
            raise HTTPException(status_code=401, detail="invalid session")
        return await _captured_response(record, submission=False)

    @app.get("/__test/state", tags=["Test Control"])
    async def get_test_state() -> dict[str, object]:
        """Return the secret-free task, scenario, session-count, and journal view."""

        return mock_state.snapshot()

    @app.post("/__test/reset", tags=["Test Control"])
    async def reset_test_state() -> SuccessStatus:
        """Clear sessions and captured traffic and restore all local defaults."""

        mock_state.reset()
        return SuccessStatus(status=True, description="mock state reset")

    @app.put("/__test/task", tags=["Test Control"])
    async def configure_task(task: TaskControlRequest) -> SuccessStatus:
        """Change the active task template used for discovery and submissions."""

        template = ApiClientTaskTemplateInfo(
            name=task.name,
            taskGroup=task.task_group,
            taskType=task.task_type,
            duration=task.duration,
        )
        mock_state.set_task(active=task.active, task=template)
        return SuccessStatus(status=True, description="task configured")

    @app.put("/__test/scenario/submission", tags=["Test Control"])
    async def configure_submission_scenario(
        scenario: SubmissionScenarioRequest,
    ) -> SuccessStatus:
        """Configure the response consumed by the next valid submission."""

        mock_state.set_scenario(
            "SUBMISSION",
            ResponseScenario(
                status_code=scenario.status_code,
                verdict=scenario.verdict,
                delay_ms=scenario.delay_ms,
                malformed_body=scenario.malformed_body,
                description=scenario.description,
            ),
        )
        return SuccessStatus(status=True, description="submission scenario configured")

    @app.put("/__test/scenario/result-log", tags=["Test Control"])
    async def configure_result_log_scenario(
        scenario: ResultLogScenarioRequest,
    ) -> SuccessStatus:
        """Configure the response consumed by the next valid result log."""

        mock_state.set_scenario(
            "RESULT_LOG",
            ResponseScenario(
                status_code=scenario.status_code,
                verdict=DresVerdict.INDETERMINATE,
                delay_ms=scenario.delay_ms,
                malformed_body=scenario.malformed_body,
                description=scenario.description,
            ),
        )
        return SuccessStatus(status=True, description="result-log scenario configured")

    @app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
    async def monitor_page() -> HTMLResponse:
        """Serve the package-owned local traffic monitor HTML page."""

        page = files("dres_mock_server").joinpath("static", "monitor.html").read_text(
            encoding="utf-8"
        )
        return HTMLResponse(content=page)

    @app.get("/static/{asset_name}", include_in_schema=False)
    async def monitor_asset(asset_name: str) -> Response:
        """Serve only explicitly named monitor assets from package resources."""

        asset = _STATIC_ASSETS.get(asset_name)
        if asset is None:
            raise HTTPException(status_code=404, detail="asset not found")
        media_type, resource_name = asset
        content = files("dres_mock_server").joinpath("static", resource_name).read_text(
            encoding="utf-8"
        )
        return Response(content=content, media_type=media_type)

    return app


def _require_session(state: MockState, session_id: str | None) -> str:
    """Return the username for a valid token or raise a generic DRES 401."""

    if not session_id:
        raise HTTPException(status_code=401, detail="invalid session")
    username = state.session_username(session_id)
    if username is None:
        raise HTTPException(status_code=401, detail="invalid session")
    return username


def _error_response(status_code: int, description: str) -> JSONResponse:
    """Serialize a DRES ErrorStatus without preserving arbitrary error objects."""

    body = ErrorStatus(status=False, description=description)
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
    )


async def _captured_response(record: CapturedRequest, *, submission: bool) -> JSONResponse:
    """Wait for the configured delay and serialize its DRES-shaped outcome."""

    outcome = record.outcome
    if outcome.delay_ms:
        # State consumes the scenario and journals the request before the wait,
        # matching the ambiguous-delivery case clients see after a timeout.
        await asyncio.sleep(outcome.delay_ms / 1_000)

    if outcome.malformed_body:
        body: dict[str, object] = {"status": True}
        if outcome.description:
            body["description"] = outcome.description
        return JSONResponse(status_code=outcome.status_code, content=body)

    if outcome.status_code in (200, 202):
        if submission:
            response = SuccessfulSubmissionsStatus(
                status=True,
                submission=outcome.verdict,
                description=outcome.description,
            )
        else:
            response = SuccessStatus(status=True, description=outcome.description)
        return JSONResponse(
            status_code=outcome.status_code,
            content=response.model_dump(mode="json"),
        )

    error = ErrorStatus(status=False, description=outcome.description)
    return JSONResponse(
        status_code=outcome.status_code,
        content=error.model_dump(mode="json"),
    )


def run() -> None:
    """Run the local mock process without logging session query parameters."""

    settings = MockSettings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        access_log=False,
    )


__all__ = ["create_app", "run"]
