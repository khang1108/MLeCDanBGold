"""Service orchestrating Query Hypothesis sessions, preview, commit, and monotonic undo."""

from __future__ import annotations

import time
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from hcmai.kis.hypothesis.logging import (
    extract_action_metadata,
    log_query_hypothesis_event,
)
from hcmai.kis.hypothesis.models import (
    QueryHypothesisAction,
    QueryHypothesisCheckpoint,
    QueryHypothesisError,
    QueryHypothesisPreview,
    QueryHypothesisSession,
    QueryHypothesisView,
)
from hcmai.kis.hypothesis.mutations import apply_query_action, restore_query_state
from hcmai.kis.hypothesis.store import QueryHypothesisStore
from hcmai.kis.models import KISImageRef


class QueryHypothesisService:
    """Manages Query Hypothesis lifecycle: open, preview, commit, undo with revision guards."""

    def __init__(
        self,
        store_or_resolver: Any = None,
        resolver_or_store: Any = None,
        image_canonicalizer: Any = None,
        clock: Callable[[], float] = time.time,
        *,
        resolver: Any = None,
        store: QueryHypothesisStore | None = None,
    ) -> None:
        if resolver is not None:
            self._resolver = resolver
        elif isinstance(store_or_resolver, QueryHypothesisStore):
            self._resolver = resolver_or_store
        else:
            self._resolver = store_or_resolver

        if store is not None:
            self._store = store
        elif isinstance(store_or_resolver, QueryHypothesisStore):
            self._store = store_or_resolver
        else:
            self._store = resolver_or_store

        if self._store is None:
            raise ValueError("QueryHypothesisStore is required")
        if self._resolver is None:
            raise ValueError("Intent resolver is required")

        self._image_canonicalizer = image_canonicalizer
        self._clock = clock

    def _require_revision(
        self, session: QueryHypothesisSession, expected_revision: int
    ) -> None:
        if session.intent.revision != expected_revision:
            raise QueryHypothesisError(
                "QUERY_REVISION_CONFLICT",
                f"expected query revision {expected_revision}, current is {session.intent.revision}",
            )

    def open(
        self, text: str, image_refs: tuple[KISImageRef, ...] = ()
    ) -> QueryHypothesisView:
        """Open a new Query Hypothesis session from initial natural text and images."""
        started = perf_counter()
        intent = self._resolver.resolve_initial(text, revision=1)
        if image_refs:
            first = intent.events[0].model_copy(update={"images": list(image_refs)})
            intent = intent.model_copy(update={"events": [first, *intent.events[1:]]})

        now = self._clock()
        session = QueryHypothesisSession(
            session_id=f"qh_{uuid4().hex}",
            intent=intent,
            history=(),
            created_at=now,
            updated_at=now,
        )
        self._store.put(session)
        total_ms = (perf_counter() - started) * 1000.0
        log_query_hypothesis_event(
            event_type="query_hypothesis_open",
            session_id=session.session_id,
            query_revision=1,
            action_type="open",
            affected_event_ids=[e.id for e in intent.events],
            latency_ms=total_ms,
            committed=True,
        )
        return QueryHypothesisView.from_session(session)

    def get(self, session_id: str) -> QueryHypothesisView:
        """Get the current view of a session."""
        return QueryHypothesisView.from_session(self._store.get(session_id))

    def preview(
        self,
        session_id: str,
        expected_revision: int,
        action: QueryHypothesisAction,
    ) -> QueryHypothesisPreview:
        """Preview the result of an action without mutating the canonical session."""
        started = perf_counter()
        action_type, affected_ids, action_payload = extract_action_metadata(action)
        with self._store.locked(session_id) as slot:
            self._require_revision(slot.session, expected_revision)
            proposed = apply_query_action(slot.session.intent, action)
            total_ms = (perf_counter() - started) * 1000.0
            log_query_hypothesis_event(
                event_type="query_hypothesis_preview",
                session_id=session_id,
                query_revision=expected_revision,
                action_type=action_type,
                affected_event_ids=affected_ids,
                latency_ms=total_ms,
                committed=False,
                action_payload=action_payload,
            )
            return QueryHypothesisPreview(
                base_revision=expected_revision, intent=proposed
            )

    def commit(
        self,
        session_id: str,
        expected_revision: int,
        action: QueryHypothesisAction,
    ) -> QueryHypothesisView:
        """Apply an action, push checkpoint to history, bump revision, and update session."""
        started = perf_counter()
        action_type, affected_ids, action_payload = extract_action_metadata(action)
        with self._store.locked(session_id) as slot:
            self._require_revision(slot.session, expected_revision)
            previous = slot.session.intent
            committed = apply_query_action(previous, action)
            slot.session.history = (
                *slot.session.history[-19:],
                QueryHypothesisCheckpoint(intent=previous),
            )
            slot.session.intent = committed
            slot.session.updated_at = self._clock()
            total_ms = (perf_counter() - started) * 1000.0
            log_query_hypothesis_event(
                event_type="query_hypothesis_commit",
                session_id=session_id,
                query_revision=committed.revision,
                action_type=action_type,
                affected_event_ids=affected_ids,
                latency_ms=total_ms,
                committed=True,
                action_payload=action_payload,
            )
            return QueryHypothesisView.from_session(slot.session)

    def undo(
        self, session_id: str, expected_revision: int
    ) -> QueryHypothesisView:
        """Pop the last checkpoint, restoring earlier intent with monotonic revision."""
        started = perf_counter()
        with self._store.locked(session_id) as slot:
            self._require_revision(slot.session, expected_revision)
            if not slot.session.history:
                raise QueryHypothesisError(
                    "QUERY_CANNOT_UNDO", "No query hypothesis checkpoint is available"
                )
            checkpoint = slot.session.history[-1]
            slot.session.history = slot.session.history[:-1]
            slot.session.intent = restore_query_state(
                slot.session.intent, checkpoint.intent
            )
            slot.session.updated_at = self._clock()
            total_ms = (perf_counter() - started) * 1000.0
            log_query_hypothesis_event(
                event_type="query_hypothesis_undo",
                session_id=session_id,
                query_revision=slot.session.intent.revision,
                action_type="undo",
                affected_event_ids=[],
                latency_ms=total_ms,
                committed=True,
            )
            return QueryHypothesisView.from_session(slot.session)
