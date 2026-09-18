"""Transactional service executing KIS chat feedback, query repair, and retrieval refinements."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from hcmai.api.contracts.feedback import (
    FeedbackOpenRequest,
    FeedbackStateResponse,
    FeedbackTurnRequest,
    FeedbackUndoRequest,
    RetrievalOverride,
)
from hcmai.api.contracts.kis import KISSearchResult
from hcmai.kis.feedback.models import (
    FeedbackAction,
    FeedbackChatTurn,
    FeedbackResolveContext,
    FeedbackSession,
    QueryEditProposalAction,
    RefineRetrievalAction,
    RejectCandidateAction,
    RepairEventAction,
)
from hcmai.kis.feedback.resolver import FeedbackResolver
from hcmai.kis.feedback.store import (
    FeedbackError,
    FeedbackSessionStore,
    commit_feedback_turn,
    undo_feedback,
)
from hcmai.kis.models import (
    KISEvent,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.retrieval.plan import KISRetrievalPlan, build_retrieval_plan


class FeedbackService:
    """Orchestrates bounded KIS feedback sessions, action resolution, and retrieval execution."""

    def __init__(
        self,
        store: FeedbackSessionStore,
        resolver: FeedbackResolver,
        search_service: Any,
        event_trail_service: Any | None = None,
    ) -> None:
        self._store = store
        self._resolver = resolver
        self._search_service = search_service
        self._event_trail_service = event_trail_service

    def open(self, request: FeedbackOpenRequest) -> FeedbackStateResponse:
        """Open a new bounded feedback session from an existing search result."""
        # Validate snapshot reference if store exists on search service
        if hasattr(self._search_service, "event_trail_snapshots"):
            try:
                self._search_service.event_trail_snapshots.get(request.evidence_snapshot_id)
            except Exception as exc:
                raise FeedbackError(
                    "SNAPSHOT_NOT_FOUND",
                    f"Evidence snapshot {request.evidence_snapshot_id} not found: {exc}",
                ) from exc

        session_id = f"fbs_{uuid4().hex[:12]}"
        scope = "all_videos" if len(request.intent.events) == 1 else "all_videos"

        initial_state = FeedbackStateResponse(
            session_id=session_id,
            status="applied",
            feedback_revision=1,
            intent=request.intent,
            retrieval_overrides={},
            results=list(request.initial_results),
            evidence_snapshot_id=request.evidence_snapshot_id,
            trail=None,
            assistant_message="Feedback session opened.",
            changed_event_ids=[],
            scope=scope,
            can_undo=False,
        )

        session = FeedbackSession(
            session_id=session_id,
            original_query=request.original_query,
            use_dense=request.use_dense,
            use_bm25=request.use_bm25,
            top_k=request.top_k,
            feedback_revision=1,
            state=initial_state,
            history=(),
            chat_turns=(),
            exclusions=(),
            committed_requests={},
        )
        self._store.put(session)
        return initial_state

    def turn(self, session_id: str, request: FeedbackTurnRequest) -> FeedbackStateResponse:
        """Execute one feedback turn under session lock."""
        with self._store.locked(session_id) as slot:
            session = slot.session

            # Check duplicate request_id idempotency
            if request.request_id in session.committed_requests:
                cached_payload, cached_response = session.committed_requests[request.request_id]
                if cached_payload == request.model_dump():
                    return cached_response
                raise FeedbackError(
                    "REQUEST_CONFLICT",
                    f"Duplicate request_id {request.request_id} with conflicting payload",
                )

            # Check revisions
            if request.expected_feedback_revision != session.feedback_revision:
                raise FeedbackError(
                    "STALE_REVISION",
                    f"Expected feedback_revision {request.expected_feedback_revision}, "
                    f"current is {session.feedback_revision}",
                )
            if request.expected_kis_revision != session.state.intent.revision:
                raise FeedbackError(
                    "STALE_REVISION",
                    f"Expected kis_revision {request.expected_kis_revision}, "
                    f"current is {session.state.intent.revision}",
                )
            if (
                request.expected_trail_revision is not None
                and session.state.trail is not None
                and request.expected_trail_revision != session.state.trail.trail_revision
            ):
                raise FeedbackError(
                    "STALE_REVISION",
                    f"Expected trail_revision {request.expected_trail_revision}, "
                    f"current is {session.state.trail.trail_revision}",
                )

            # Build context for resolver
            context = FeedbackResolveContext(
                original_query=session.original_query,
                intent=session.state.intent,
                retrieval_overrides=session.state.retrieval_overrides,
                selected_result_id=request.selected_result_id,
                selected_event_id=request.selected_event_id,
                selected_frame_id=request.selected_frame_id,
                anchors={},
                recent_turns=list(session.chat_turns[-4:]),
                scope=session.state.scope,
            )

            action = self._resolver.resolve(context, request.message)

            # Record chat turns
            user_turn = FeedbackChatTurn(
                role="user",
                message=request.message,
                timestamp_ms=perf_counter() * 1000,
            )
            session.chat_turns = (*session.chat_turns, user_turn)

            # Execute action
            if action.type == "clarify":
                assistant_turn = FeedbackChatTurn(
                    role="assistant",
                    message=action.question,
                    action_type="clarify",
                    timestamp_ms=perf_counter() * 1000,
                )
                session.chat_turns = (*session.chat_turns, assistant_turn)
                clarify_state = session.state.model_copy(
                    update={
                        "status": "clarification",
                        "assistant_message": action.question,
                        "changed_event_ids": [],
                    }
                )
                session.committed_requests[request.request_id] = (
                    request.model_dump(),
                    clarify_state,
                )
                return clarify_state

            if action.type == "refine_retrieval":
                overrides = dict(session.state.retrieval_overrides)
                for eid, text in action.refinements.items():
                    overrides[eid] = RetrievalOverride(dense_text=text, bm25_text=text)

                search_exec = self._execute_search(
                    intent=session.state.intent,
                    overrides=overrides,
                    use_dense=session.use_dense,
                    use_bm25=session.use_bm25,
                    top_k=session.top_k,
                    exclusions=session.exclusions,
                )
                next_feedback_rev = session.feedback_revision + 1
                new_state = FeedbackStateResponse(
                    session_id=session_id,
                    status="applied",
                    feedback_revision=next_feedback_rev,
                    intent=session.state.intent,
                    retrieval_overrides=overrides,
                    results=search_exec.results,
                    evidence_snapshot_id=search_exec.evidence_snapshot_id,
                    trail=session.state.trail,
                    assistant_message=f"Refined search query for {', '.join(action.event_ids)}.",
                    changed_event_ids=list(action.event_ids),
                    scope=session.state.scope,
                    can_undo=True,
                    latency=getattr(search_exec, "latency", None),
                )
                return commit_feedback_turn(session, request, new_state)

            if action.type == "query_edit_proposal":
                assistant_turn = FeedbackChatTurn(
                    role="assistant",
                    message=action.explanation,
                    action_type="query_edit_proposal",
                    timestamp_ms=perf_counter() * 1000,
                )
                session.chat_turns = (*session.chat_turns, assistant_turn)
                proposal_state = session.state.model_copy(
                    update={
                        "status": "proposal",
                        "assistant_message": action.explanation,
                        "query_proposal": action.model_dump(),
                        "changed_event_ids": [],
                        "can_undo": session.state.can_undo,
                    }
                )
                session.committed_requests[request.request_id] = (
                    request.model_dump(),
                    proposal_state,
                )
                return proposal_state

            if action.type == "edit_intent":
                eid = action.event_ids[0] if action.event_ids else "E1"
                rep_text = action.replacement_texts.get(eid, "")
                prop = QueryEditProposalAction(
                    action={"type": "edit", "event_id": eid, "text": rep_text},
                    explanation=f"Proposed edit for {eid}",
                )
                assistant_turn = FeedbackChatTurn(
                    role="assistant",
                    message=prop.explanation,
                    action_type="query_edit_proposal",
                    timestamp_ms=perf_counter() * 1000,
                )
                session.chat_turns = (*session.chat_turns, assistant_turn)
                proposal_state = session.state.model_copy(
                    update={
                        "status": "proposal",
                        "assistant_message": prop.explanation,
                        "query_proposal": prop.model_dump(),
                        "changed_event_ids": [],
                        "can_undo": session.state.can_undo,
                    }
                )
                session.committed_requests[request.request_id] = (
                    request.model_dump(),
                    proposal_state,
                )
                return proposal_state

            if action.type == "restructure":
                prop = QueryEditProposalAction(
                    action={
                        "type": "split",
                        "event_id": action.replaced_event_ids[0] if action.replaced_event_ids else "E1",
                        "new_events": action.new_events,
                    },
                    explanation="Proposed event restructure",
                )
                assistant_turn = FeedbackChatTurn(
                    role="assistant",
                    message=prop.explanation,
                    action_type="query_edit_proposal",
                    timestamp_ms=perf_counter() * 1000,
                )
                session.chat_turns = (*session.chat_turns, assistant_turn)
                proposal_state = session.state.model_copy(
                    update={
                        "status": "proposal",
                        "assistant_message": prop.explanation,
                        "query_proposal": prop.model_dump(),
                        "changed_event_ids": [],
                        "can_undo": session.state.can_undo,
                    }
                )
                session.committed_requests[request.request_id] = (
                    request.model_dump(),
                    proposal_state,
                )
                return proposal_state

            if action.type == "reject_candidate":
                target_eid = action.event_id
                candidate_frame = (
                    action.candidate_frame_id or request.selected_frame_id or ""
                )
                selected_result = None
                for r in session.state.results:
                    if r.result_id == request.selected_result_id:
                        selected_result = r
                        break

                if selected_result is not None:
                    video_id = selected_result.video_id
                elif hasattr(self._search_service, "event_trail_snapshots"):
                    try:
                        snap = self._search_service.event_trail_snapshots.get(
                            session.state.evidence_snapshot_id
                        )
                        if (
                            hasattr(snap, "results")
                            and request.selected_result_id in snap.results
                        ):
                            video_id = snap.results[request.selected_result_id].video_id
                        else:
                            video_id = "unknown"
                    except Exception:
                        video_id = "unknown"
                else:
                    video_id = "unknown"

                exclusion = (video_id, target_eid, candidate_frame)
                new_exclusions = (*session.exclusions, exclusion)

                search_exec = self._execute_search(
                    intent=session.state.intent,
                    overrides=session.state.retrieval_overrides,
                    use_dense=session.use_dense,
                    use_bm25=session.use_bm25,
                    top_k=session.top_k,
                    exclusions=new_exclusions,
                )
                next_feedback_rev = session.feedback_revision + 1
                new_state = FeedbackStateResponse(
                    session_id=session_id,
                    status="applied",
                    feedback_revision=next_feedback_rev,
                    intent=session.state.intent,
                    retrieval_overrides=session.state.retrieval_overrides,
                    results=search_exec.results,
                    evidence_snapshot_id=search_exec.evidence_snapshot_id,
                    trail=session.state.trail,
                    assistant_message=f"Excluded candidate frame for {target_eid}.",
                    changed_event_ids=[target_eid],
                    scope=session.state.scope,
                    can_undo=True,
                    latency=getattr(search_exec, "latency", None),
                )
                return commit_feedback_turn(
                    session, request, new_state, exclusions=new_exclusions
                )

            if action.type == "repair_event":
                target_eid = action.event_id
                overrides = dict(session.state.retrieval_overrides)
                # If message contains descriptive clues, use them as visual keyword refinement
                if request.message and len(request.message.strip()) > 0:
                    overrides[target_eid] = RetrievalOverride(
                        dense_text=request.message.strip(),
                        bm25_text=request.message.strip(),
                    )

                search_exec = self._execute_search(
                    intent=session.state.intent,
                    overrides=overrides,
                    use_dense=session.use_dense,
                    use_bm25=session.use_bm25,
                    top_k=session.top_k,
                    exclusions=session.exclusions,
                )
                next_feedback_rev = session.feedback_revision + 1
                new_state = FeedbackStateResponse(
                    session_id=session_id,
                    status="applied",
                    feedback_revision=next_feedback_rev,
                    intent=session.state.intent,
                    retrieval_overrides=overrides,
                    results=search_exec.results,
                    evidence_snapshot_id=search_exec.evidence_snapshot_id,
                    trail=session.state.trail,
                    assistant_message=f"Re-searched event {target_eid}.",
                    changed_event_ids=[target_eid],
                    scope=session.state.scope,
                    can_undo=True,
                    latency=getattr(search_exec, "latency", None),
                )
                return commit_feedback_turn(session, request, new_state)

            # Fallback for anchor or other actions
            search_exec = self._execute_search(
                intent=session.state.intent,
                overrides=session.state.retrieval_overrides,
                use_dense=session.use_dense,
                use_bm25=session.use_bm25,
                top_k=session.top_k,
                exclusions=session.exclusions,
            )
            next_feedback_rev = session.feedback_revision + 1
            new_state = FeedbackStateResponse(
                session_id=session_id,
                status="applied",
                feedback_revision=next_feedback_rev,
                intent=session.state.intent,
                retrieval_overrides=session.state.retrieval_overrides,
                results=search_exec.results,
                evidence_snapshot_id=search_exec.evidence_snapshot_id,
                trail=session.state.trail,
                assistant_message="Updated results.",
                changed_event_ids=[getattr(action, "event_id", "")],
                scope=session.state.scope,
                can_undo=True,
                latency=getattr(search_exec, "latency", None),
            )
            return commit_feedback_turn(session, request, new_state)

    def undo(self, session_id: str, request: FeedbackUndoRequest) -> FeedbackStateResponse:
        """Undo the most recent committed feedback turn."""
        with self._store.locked(session_id) as slot:
            session = slot.session
            if request.expected_feedback_revision != session.feedback_revision:
                raise FeedbackError(
                    "STALE_REVISION",
                    f"Expected feedback_revision {request.expected_feedback_revision}, "
                    f"current is {session.feedback_revision}",
                )
            return undo_feedback(session)

    def _execute_search(
        self,
        intent: KISIntent,
        overrides: dict[str, RetrievalOverride],
        use_dense: bool,
        use_bm25: bool,
        top_k: int,
        exclusions: tuple[tuple[str, str, str], ...],
    ) -> Any:
        plan = build_retrieval_plan(
            intent,
            overrides=overrides,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )

        if hasattr(self._search_service, "execute_search"):
            return self._search_service.execute_search(
                intent=intent,
                retrieval_plan=plan,
                use_dense=use_dense,
                use_bm25=use_bm25,
                top_k=top_k,
                exclusions=exclusions,
            )

        # Real SearchService integration
        execution = self._search_service.kis.execute(
            intent=intent,
            retrieval_plan=plan,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=top_k,
        )
        result_ids = [f"r_{uuid4().hex}" for _ in execution.results]
        kis_results = [
            KISSearchResult(result_id=rid, **res.model_dump())
            for rid, res in zip(result_ids, execution.results, strict=True)
        ]

        # Filter excluded candidate frames/occurrences
        if exclusions:
            filtered = []
            for r in kis_results:
                is_excluded = any(
                    r.video_id == ex[0] and (str(r.frame_idx) == str(ex[2]) or str(r.timestamp_ms) == str(ex[2]))
                    for ex in exclusions
                )
                if not is_excluded:
                    filtered.append(r)
            kis_results = filtered

        # Snapshot temporal evidence for EventTrail
        snap_id = None
        snapshot_ms = 0.0
        if hasattr(self._search_service, "create_evidence_snapshot"):
            snap_id, snapshot_ms, _ = self._search_service.create_evidence_snapshot(
                intent=intent,
                execution=execution,
                result_ids=result_ids,
            )
        if not snap_id:
            snap_id = getattr(execution, "evidence_snapshot_id", None) or f"snap_{uuid4().hex[:12]}"

        latency_dict = None
        if hasattr(execution, "latency"):
            lat = execution.latency
            if hasattr(lat, "model_dump"):
                latency_dict = lat.model_dump()
            elif isinstance(lat, dict):
                latency_dict = dict(lat)
            if latency_dict and snapshot_ms > 0:
                latency_dict["snapshot_ms"] = snapshot_ms
                latency_dict["total_ms"] = latency_dict.get("total_ms", 0.0) + snapshot_ms
        elif hasattr(execution, "results"):
            latency_dict = {"total_ms": snapshot_ms, "snapshot_ms": snapshot_ms}

        return type(
            "SearchExecutionResult",
            (),
            {
                "results": kis_results,
                "evidence_snapshot_id": snap_id,
                "latency": latency_dict,
            },
        )()


__all__ = ["FeedbackService"]
