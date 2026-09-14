"""Thread-safe in-memory state for the standalone DRES mock server.

This module owns local sessions, active task metadata, one-shot response
scenarios, and bounded request journals. It does not implement HTTP routes or
semantic scoring; captured payloads retain DRES aliases and supplied fields.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

from dres_mock_server.models import (
    ApiClientSubmission,
    ApiClientTaskTemplateInfo,
    DresVerdict,
    QueryResultLog,
)
from dres_mock_server.settings import MockSettings


RequestKind = Literal["SUBMISSION", "RESULT_LOG"]
_JOURNAL_LIMIT = 1_000
_SCENARIO_STATUS_CODES = {200, 202, 400, 401, 404, 412, 500}


@dataclass(frozen=True)
class _Session:
    """A private mapping value whose opaque token is never shown in repr."""

    username: str
    token: str = field(repr=False)


@dataclass(frozen=True)
class ResponseScenario:
    """One response configuration consumed by the next matching request."""

    status_code: Literal[200, 202, 400, 401, 404, 412, 500] = 200
    verdict: DresVerdict = DresVerdict.INDETERMINATE
    delay_ms: int = 0
    malformed_body: bool = False
    description: str = "mock accepted"

    def __post_init__(self) -> None:
        """Reject invalid scenarios even when they bypass HTTP control models."""

        if self.status_code not in _SCENARIO_STATUS_CODES:
            raise ValueError("unsupported mock response status code")
        if isinstance(self.delay_ms, bool) or not 0 <= self.delay_ms <= 30_000:
            raise ValueError("scenario delay must be between 0 and 30000 ms")
        if not isinstance(self.description, str):
            raise TypeError("scenario description must be a string")
        if not isinstance(self.malformed_body, bool):
            raise TypeError("malformed_body must be a boolean")
        if not isinstance(self.verdict, DresVerdict):
            object.__setattr__(self, "verdict", DresVerdict(self.verdict))


@dataclass(frozen=True)
class CapturedOutcome:
    """Response selected for a captured request, without caller secrets."""

    status_code: int
    verdict: DresVerdict | None
    description: str
    delay_ms: int = 0
    malformed_body: bool = False


@dataclass(frozen=True)
class CapturedRequest:
    """A validated request record with actor identity but no session token."""

    record_id: int
    kind: RequestKind
    username: str
    session_fingerprint: str
    evaluation_id: str
    received_at_ms: int
    payload: dict[str, object]
    outcome: CapturedOutcome


class MockState:
    """Own synchronized mock sessions, task/scenario settings, and journals."""

    def __init__(self, settings: MockSettings) -> None:
        """Initialize private state and restore configured defaults."""

        self._settings = settings
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}
        self._journals: dict[RequestKind, list[CapturedRequest]] = {
            "SUBMISSION": [],
            "RESULT_LOG": [],
        }
        self._next_record_id = 1
        self._task_template = self._default_task()
        self._task_is_active = True
        self._scenarios: dict[RequestKind, ResponseScenario] = {}
        self._restore_default_scenarios()

    def login(self, username: str, password: str) -> str | None:
        """Validate configured credentials and create a new opaque session."""

        with self._lock:
            if not self._settings.credential_matches(username, password):
                return None
            return self._create_session_locked(username)

    def create_session(self, username: str) -> str:
        """Create a session for credentials the caller has already validated."""

        with self._lock:
            return self._create_session_locked(username)

    def session_username(self, session_id: str | None) -> str | None:
        """Return the authenticated username for an active token, if any."""

        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            return session.username if session is not None else None

    def logout(self, session_id: str | None) -> bool:
        """Invalidate only the selected session and report whether it existed."""

        if not session_id:
            return False
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    @property
    def session_count(self) -> int:
        """Return the current session count without exposing session tokens."""

        with self._lock:
            return len(self._sessions)

    def active_task(self) -> ApiClientTaskTemplateInfo | None:
        """Return a detached active-task model or ``None`` when taskless."""

        with self._lock:
            if not self._task_is_active:
                return None
            return self._copy_task(self._task_template)

    def set_active_task(self, task: ApiClientTaskTemplateInfo | None) -> None:
        """Set task metadata, or disable the task while retaining its metadata."""

        with self._lock:
            if task is None:
                self._task_is_active = False
                return
            self._task_template = self._copy_task(task)
            self._task_is_active = True

    def set_task(
        self,
        active: bool,
        task: ApiClientTaskTemplateInfo | None = None,
    ) -> None:
        """Update task activation and optional template metadata together."""

        if not isinstance(active, bool):
            raise TypeError("active must be a boolean")
        with self._lock:
            if task is not None:
                self._task_template = self._copy_task(task)
            self._task_is_active = active

    def set_scenario(self, kind: RequestKind, scenario: ResponseScenario) -> None:
        """Replace the one-shot scenario for submissions or result logs."""

        self._validate_kind(kind)
        if not isinstance(scenario, ResponseScenario):
            raise TypeError("scenario must be a ResponseScenario")
        with self._lock:
            self._scenarios[kind] = scenario

    def capture_submission(
        self,
        session_id: str | None,
        evaluation_id: str,
        submission: ApiClientSubmission,
    ) -> CapturedRequest | None:
        """Authenticate, resolve the active task, record, and select a response."""

        payload = self._payload_from_model(submission)
        with self._lock:
            session = self._sessions.get(session_id or "")
            if session is None:
                return None

            rejection = self._submission_task_rejection(submission)
            if rejection is not None:
                outcome = CapturedOutcome(
                    status_code=412,
                    verdict=None,
                    description=rejection,
                )
            else:
                scenario = self._consume_scenario_locked("SUBMISSION")
                outcome = self._outcome_from_scenario(scenario, include_verdict=True)

            return self._append_record_locked(
                kind="SUBMISSION",
                session_id=session_id or "",
                session=session,
                evaluation_id=evaluation_id,
                payload=payload,
                outcome=outcome,
            )

    def capture_result_log(
        self,
        session_id: str | None,
        evaluation_id: str,
        result_log: QueryResultLog,
    ) -> CapturedRequest | None:
        """Authenticate, record a result log, and atomically consume its scenario."""

        payload = self._payload_from_model(result_log)
        with self._lock:
            session = self._sessions.get(session_id or "")
            if session is None:
                return None

            scenario = self._consume_scenario_locked("RESULT_LOG")
            outcome = self._outcome_from_scenario(scenario, include_verdict=False)
            return self._append_record_locked(
                kind="RESULT_LOG",
                session_id=session_id or "",
                session=session,
                evaluation_id=evaluation_id,
                payload=payload,
                outcome=outcome,
            )

    def captured_requests(self, kind: RequestKind | None = None) -> list[CapturedRequest]:
        """Return detached records, optionally restricted to one journal."""

        if kind is not None:
            self._validate_kind(kind)
        with self._lock:
            if kind is not None:
                records = self._journals[kind]
            else:
                records = sorted(
                    self._journals["SUBMISSION"] + self._journals["RESULT_LOG"],
                    key=lambda record: record.record_id,
                )
            return [self._copy_record(record) for record in records]

    def snapshot(self) -> dict[str, object]:
        """Return a secret-free, JSON-ready view for the local test controls."""

        with self._lock:
            submissions = [
                self._record_to_dict(record)
                for record in reversed(self._journals["SUBMISSION"])
            ]
            result_logs = [
                self._record_to_dict(record)
                for record in reversed(self._journals["RESULT_LOG"])
            ]
            task = self._task_template.model_dump(
                mode="json",
                by_alias=True,
                exclude_unset=True,
            )
            task["active"] = self._task_is_active
            scenarios = {
                "submission": self._scenario_to_dict(self._scenarios["SUBMISSION"]),
                "result_log": self._scenario_to_dict(self._scenarios["RESULT_LOG"]),
            }

            return {
                "evaluation": {
                    "id": self._settings.evaluation_id,
                    "name": self._settings.evaluation_name,
                    "type": self._settings.evaluation_type,
                },
                "task": task,
                "session_count": len(self._sessions),
                "scenarios": scenarios,
                "submissions": submissions,
                "result_logs": result_logs,
                "submission_count": len(self._journals["SUBMISSION"]),
                "result_log_count": len(self._journals["RESULT_LOG"]),
            }

    def reset(self) -> None:
        """Clear sessions and traffic and restore task, scenarios, and IDs."""

        with self._lock:
            self._sessions.clear()
            self._journals["SUBMISSION"].clear()
            self._journals["RESULT_LOG"].clear()
            self._next_record_id = 1
            self._task_template = self._default_task()
            self._task_is_active = True
            self._restore_default_scenarios()

    def _create_session_locked(self, username: str) -> str:
        """Create an opaque token while the caller owns the state lock."""

        while True:
            token = secrets.token_urlsafe(32)
            if token not in self._sessions:
                break
        self._sessions[token] = _Session(username=username, token=token)
        return token

    def _append_record_locked(
        self,
        *,
        kind: RequestKind,
        session_id: str,
        session: _Session,
        evaluation_id: str,
        payload: dict[str, object],
        outcome: CapturedOutcome,
    ) -> CapturedRequest:
        """Append a record and trim only the selected bounded journal."""

        fingerprint = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
        record = CapturedRequest(
            record_id=self._next_record_id,
            kind=kind,
            username=session.username,
            session_fingerprint=fingerprint,
            evaluation_id=evaluation_id,
            received_at_ms=time.time_ns() // 1_000_000,
            payload=payload,
            outcome=outcome,
        )
        self._next_record_id += 1

        journal = self._journals[kind]
        journal.append(record)
        if len(journal) > _JOURNAL_LIMIT:
            del journal[: len(journal) - _JOURNAL_LIMIT]
        return self._copy_record(record)

    def _submission_task_rejection(
        self,
        submission: ApiClientSubmission,
    ) -> str | None:
        """Return a DRES-like rejection description before consuming a scenario."""

        if not self._task_is_active:
            return "No active task is available."

        active_name = self._task_template.name
        for answer_set in submission.answer_sets:
            if "task_id" in answer_set.model_fields_set:
                return "This mock has no task instance ID to match."
            if answer_set.task_name is not None and answer_set.task_name != active_name:
                return "Submission task name does not match the active task."
        return None

    def _consume_scenario_locked(self, kind: RequestKind) -> ResponseScenario:
        """Take one configured scenario and atomically restore its default."""

        scenario = self._scenarios[kind]
        self._scenarios[kind] = self._default_scenario()
        return scenario

    @staticmethod
    def _outcome_from_scenario(
        scenario: ResponseScenario,
        *,
        include_verdict: bool,
    ) -> CapturedOutcome:
        """Copy the selected scenario into a secret-free captured outcome."""

        return CapturedOutcome(
            status_code=scenario.status_code,
            verdict=scenario.verdict if include_verdict else None,
            description=scenario.description,
            delay_ms=scenario.delay_ms,
            malformed_body=scenario.malformed_body,
        )

    def _default_task(self) -> ApiClientTaskTemplateInfo:
        """Build fresh task metadata from the application-local settings."""

        return ApiClientTaskTemplateInfo(
            name=self._settings.default_task_name,
            taskGroup=self._settings.default_task_group,
            taskType=self._settings.default_task_type,
            duration=self._settings.default_task_duration,
        )

    @staticmethod
    def _default_scenario() -> ResponseScenario:
        """Return the harmless response restored after a one-shot scenario."""

        return ResponseScenario()

    def _restore_default_scenarios(self) -> None:
        """Replace each operation's pending response with a fresh default."""

        self._scenarios = {
            "SUBMISSION": self._default_scenario(),
            "RESULT_LOG": self._default_scenario(),
        }

    @staticmethod
    def _payload_from_model(model: ApiClientSubmission | QueryResultLog) -> dict[str, object]:
        """Preserve official aliases and distinguish unset fields from explicit nulls."""

        return model.model_dump(mode="json", by_alias=True, exclude_unset=True)

    @staticmethod
    def _copy_task(task: ApiClientTaskTemplateInfo) -> ApiClientTaskTemplateInfo:
        """Return a detached template so callers cannot mutate guarded state."""

        return ApiClientTaskTemplateInfo.model_validate(
            task.model_dump(mode="python", by_alias=True, exclude_unset=True)
        )

    @staticmethod
    def _copy_record(record: CapturedRequest) -> CapturedRequest:
        """Return a detached record including a recursively copied payload."""

        return CapturedRequest(
            record_id=record.record_id,
            kind=record.kind,
            username=record.username,
            session_fingerprint=record.session_fingerprint,
            evaluation_id=record.evaluation_id,
            received_at_ms=record.received_at_ms,
            payload=deepcopy(record.payload),
            outcome=record.outcome,
        )

    @staticmethod
    def _record_to_dict(record: CapturedRequest) -> dict[str, object]:
        """Convert a captured record to a UI/API-safe JSON-ready mapping."""

        return {
            "record_id": record.record_id,
            "kind": record.kind,
            "username": record.username,
            "session_fingerprint": record.session_fingerprint,
            "evaluation_id": record.evaluation_id,
            "received_at_ms": record.received_at_ms,
            "payload": deepcopy(record.payload),
            "outcome": {
                "status_code": record.outcome.status_code,
                "verdict": (
                    record.outcome.verdict.value
                    if record.outcome.verdict is not None
                    else None
                ),
                "description": record.outcome.description,
                "delay_ms": record.outcome.delay_ms,
                "malformed_body": record.outcome.malformed_body,
            },
        }

    @staticmethod
    def _scenario_to_dict(scenario: ResponseScenario) -> dict[str, object]:
        """Render a pending response scenario with the test API's field aliases."""

        return {
            "statusCode": scenario.status_code,
            "verdict": scenario.verdict.value,
            "delayMs": scenario.delay_ms,
            "malformedBody": scenario.malformed_body,
            "description": scenario.description,
        }

    @staticmethod
    def _validate_kind(kind: str) -> None:
        """Reject unknown journal or scenario names rather than creating globals."""

        if kind not in ("SUBMISSION", "RESULT_LOG"):
            raise ValueError("kind must be 'SUBMISSION' or 'RESULT_LOG'")


__all__ = [
    "CapturedOutcome",
    "CapturedRequest",
    "MockState",
    "RequestKind",
    "ResponseScenario",
]
