"""HTTPX adapter for the minimal DRES Client API v2 used by HCMAI.

This module owns DRES paths, aliases, status translation, and safe transport
errors. Session policy and task selection belong to the VBS service layer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from hcmai.vbs.config import DresSettings
from hcmai.vbs.models import (
    ApiClientSubmission,
    DresEvaluation,
    DresSubmissionStatus,
    DresStatus,
    DresTaskTemplateInfo,
    DresUser,
    QueryResultLog,
)


class DresError(RuntimeError):
    """Base exception for a safe, typed DRES integration failure."""


class DresAuthenticationError(DresError):
    """DRES rejected credentials or an expired participant session."""


class DresNoActiveTaskError(DresError):
    """DRES has no current task for the selected evaluation."""


class DresRejectedSubmissionError(DresError):
    """DRES definitively rejected an answer or result-log request."""


class DresUnavailableError(DresError):
    """DRES could not be reached or returned an unusable response."""


class DresClient:
    """Send DRES v2 requests while keeping session values out of errors."""

    def __init__(
        self,
        settings: DresSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create one reusable HTTP client with the configured hard timeout."""

        self._settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            transport=transport,
        )

    async def login(self, username: str, password: str) -> DresUser:
        """Authenticate one participant and return its private DRES session."""

        response = await self._request(
            "POST",
            "/api/v2/login",
            operation="login",
            json={"username": username, "password": password},
            sensitive_values=(password,),
        )
        try:
            return DresUser.model_validate(response.json())
        except (ValueError, ValidationError):
            raise DresUnavailableError("DRES login returned an invalid response") from None

    async def list_evaluations(self, session_id: str) -> list[DresEvaluation]:
        """List evaluations visible to a connected DRES participant."""

        response = await self._request(
            "GET",
            "/api/v2/client/evaluation/list",
            operation="list evaluations",
            params={"session": session_id},
            sensitive_values=(session_id,),
        )
        try:
            value = response.json()
            if not isinstance(value, list):
                raise TypeError("evaluation response must be a list")
            return [DresEvaluation.model_validate(item) for item in value]
        except (ValueError, TypeError, ValidationError):
            raise DresUnavailableError("DRES evaluation list returned an invalid response") from None

    async def get_current_task(
        self,
        evaluation_id: str,
        session_id: str,
    ) -> DresTaskTemplateInfo:
        """Read official task-template metadata for one evaluation and session."""

        response = await self._request(
            "GET",
            f"/api/v2/client/evaluation/currentTask/{quote(evaluation_id, safe='')}",
            operation="get current task",
            params={"session": session_id},
            sensitive_values=(session_id,),
        )
        try:
            return DresTaskTemplateInfo.model_validate(response.json())
        except (ValueError, ValidationError):
            raise DresUnavailableError("DRES current-task response was invalid") from None

    async def submit(
        self,
        evaluation_id: str,
        session_id: str,
        submission: ApiClientSubmission,
    ) -> DresSubmissionStatus:
        """Forward one answer payload and require DRES's verdict-bearing body."""

        response = await self._request(
            "POST",
            f"/api/v2/submit/{quote(evaluation_id, safe='')}",
            operation="submit answers",
            params={"session": session_id},
            json=submission.model_dump(by_alias=True, exclude_none=True),
            sensitive_values=(session_id,),
            allow_accepted=True,
        )
        return _submission_status_body(response)

    async def log_results(
        self,
        evaluation_id: str,
        session_id: str,
        payload: QueryResultLog,
    ) -> DresStatus:
        """Post one QueryResultLog through the result endpoint only."""

        response = await self._request(
            "POST",
            f"/api/v2/log/result/{quote(evaluation_id, safe='')}",
            operation="log results",
            params={"session": session_id},
            json=payload.model_dump(by_alias=True, exclude_none=True),
            sensitive_values=(session_id,),
        )
        return _status_body(response, "result log")

    async def aclose(self) -> None:
        """Close the reusable underlying HTTPX connection pool."""

        await self._http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        params: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        sensitive_values: Sequence[str] = (),
        allow_accepted: bool = False,
    ) -> httpx.Response:
        """Make one request and convert every transport failure to safe text."""

        try:
            response = await self._http.request(
                method,
                path,
                params=params,
                json=json,
            )
        except httpx.TimeoutException:
            raise DresUnavailableError(f"DRES {operation} timed out") from None
        except httpx.TransportError:
            raise DresUnavailableError(f"DRES {operation} failed due to a network error") from None

        accepted_statuses = {200, 202} if allow_accepted else {200}
        if response.status_code in accepted_statuses:
            return response
        detail = _safe_detail(response, sensitive_values)
        if response.status_code == 401:
            raise DresAuthenticationError(f"DRES {operation} was unauthorized: {detail}")
        if response.status_code == 404 and operation == "get current task":
            raise DresNoActiveTaskError(f"DRES has no active task: {detail}")
        if response.status_code in {400, 404, 412}:
            raise DresRejectedSubmissionError(
                f"DRES rejected {operation} (HTTP {response.status_code}): {detail}"
            )
        raise DresUnavailableError(
            f"DRES {operation} failed (HTTP {response.status_code}): {detail}"
        )


def _status_body(response: httpx.Response, operation: str) -> DresStatus:
    """Parse successful status JSON, or accept an empty asynchronous 202 body."""

    if not response.content:
        return DresStatus(status=True, description=f"{operation} accepted")
    try:
        result = DresStatus.model_validate(response.json())
    except (ValueError, ValidationError):
        raise DresUnavailableError(f"DRES {operation} returned an invalid status") from None
    if not result.status:
        raise DresRejectedSubmissionError(f"DRES rejected {operation}: {_redact(result.description)}")
    return result


def _submission_status_body(response: httpx.Response) -> DresSubmissionStatus:
    """Require the official verdict body for both accepted submit statuses."""

    try:
        result = DresSubmissionStatus.model_validate(response.json())
    except (ValueError, ValidationError):
        raise DresUnavailableError(
            "DRES submission returned an invalid status"
        ) from None
    if not result.status:
        raise DresRejectedSubmissionError(
            f"DRES rejected submission: {_redact(result.description)}"
        )
    return result


def _safe_detail(response: httpx.Response, sensitive_values: Sequence[str]) -> str:
    """Extract a short DRES description while removing session/password values."""

    detail = f"DRES returned HTTP {response.status_code}"
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        body = None
    if isinstance(body, dict) and isinstance(body.get("description"), str):
        detail = body["description"][:500]
    for secret in sensitive_values:
        if secret:
            detail = detail.replace(secret, "[redacted]")
    return _redact(detail)


def _redact(value: str) -> str:
    """Redact accidental session parameters in descriptions from DRES."""

    return re.sub(
        r"(?i)(session\s*[=:]\s*)[^&\s,;]+",
        r"\1[redacted]",
        value,
    )


__all__ = [
    "DresAuthenticationError",
    "DresClient",
    "DresError",
    "DresNoActiveTaskError",
    "DresRejectedSubmissionError",
    "DresUnavailableError",
]
