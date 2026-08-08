"""Bounded HTTP adapter for the DRES client and submission API."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import httpx
from pydantic import TypeAdapter, ValidationError

from hcmai.common.schemas import (
    MiniChallengeEvaluation,
    MiniChallengeLoginRequest,
    MiniChallengeLoginResponse,
    MiniChallengeSubmission,
    MiniChallengeSubmissionResult,
    MiniChallengeTaskTemplate,
)
from hcmai.common.utils.logging import get_logger

logger = get_logger(__name__)
ResponseType = TypeVar("ResponseType")


class DRESClientError(RuntimeError):
    """Safe error propagated across the local API boundary."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class DRESClient:
    """Call one configured DRES instance without retrying submissions."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # Their INFO/DEBUG messages contain the full query string, including
        # the DRES session token. HCMAI emits its own token-free request logs.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def login(
        self, request: MiniChallengeLoginRequest
    ) -> MiniChallengeLoginResponse:
        return await self._request(
            "POST",
            "api/v2/login",
            None,
            TypeAdapter(MiniChallengeLoginResponse),
            json=request.model_dump(mode="json", by_alias=True),
        )

    async def list_evaluations(
        self, session: str
    ) -> list[MiniChallengeEvaluation]:
        return await self._request(
            "GET",
            "api/v2/client/evaluation/list",
            session,
            TypeAdapter(list[MiniChallengeEvaluation]),
        )

    async def current_task(
        self, evaluation_id: str, session: str
    ) -> MiniChallengeTaskTemplate:
        return await self._request(
            "GET",
            f"api/v2/client/evaluation/currentTask/{evaluation_id}",
            session,
            TypeAdapter(MiniChallengeTaskTemplate),
        )

    async def submit(
        self,
        evaluation_id: str,
        session: str,
        submission: MiniChallengeSubmission,
    ) -> MiniChallengeSubmissionResult:
        return await self._request(
            "POST",
            f"api/v2/submit/{evaluation_id}",
            session,
            TypeAdapter(MiniChallengeSubmissionResult),
            json=submission.model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
        )

    async def _request(
        self,
        method: str,
        path: str,
        session: str | None,
        adapter: TypeAdapter[ResponseType],
        **kwargs: Any,
    ) -> ResponseType:
        logger.info("DRES request started method=%s path=%s", method, path)
        params = kwargs.pop("params", {}) or {}
        if session is not None:
            params["session"] = session
        try:
            response = await self.client.request(
                method, path, params=params if params else None, **kwargs
            )
        except httpx.TimeoutException as error:
            logger.warning("DRES request timed out method=%s path=%s", method, path)
            raise DRESClientError("DRES request timed out", status_code=504) from error
        except httpx.RequestError as error:
            logger.warning("DRES request failed method=%s path=%s", method, path)
            raise DRESClientError("Could not reach DRES") from error

        if not response.is_success:
            description = None
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    description = payload.get("description") or payload.get("message")
            except ValueError:
                raw_text = response.text.strip()
                if raw_text:
                    description = f"DRES error ({response.status_code}): {raw_text[:200]}"
            safe_status = response.status_code if response.status_code in {
                400, 401, 403, 404, 412
            } else 502
            logger.warning(
                "DRES request rejected method=%s path=%s status=%d body=%s",
                method,
                path,
                response.status_code,
                response.text[:200],
            )
            raise DRESClientError(
                description or f"DRES rejected the request ({response.status_code})",
                status_code=safe_status,
            )

        payload = self._json(response, path)
        try:
            result = adapter.validate_python(payload)
        except ValidationError as error:
            logger.warning(
                "DRES payload validation error method=%s path=%s error=%s",
                method,
                path,
                error,
            )
            raise DRESClientError("DRES returned an invalid response contract") from error
        logger.info(
            "DRES request completed method=%s path=%s status=%d",
            method,
            path,
            response.status_code,
        )
        return result

    @staticmethod
    def _json(response: httpx.Response, path: str) -> Any:
        try:
            return response.json()
        except ValueError as error:
            logger.warning(
                "DRES returned non-JSON response status=%d path=%s text=%s",
                response.status_code,
                path,
                response.text[:200],
            )
            raise DRESClientError(
                f"DRES returned invalid JSON for {path} (status {response.status_code})"
            ) from error

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

