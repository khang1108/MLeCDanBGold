"""Conversational Known-Item Search (KISC) state manager.

Manages stateful conversation turns, human frame feedback (accepted and
rejected frame IDs), and applies feedback filtering to search responses.
"""

from __future__ import annotations

import time
import uuid

from hcmai.common.schemas import (
    ConversationSession,
    ConversationTurn,
    FrameFeedback,
    SearchRequest,
    SearchResponse,
    SubmissionResult,
)
from hcmai.data import FrameStore
from hcmai.search import SearchEngine


class KiscSessionManager:
    """In-memory session manager for KISC conversational searches."""

    def __init__(self) -> None:
        """Initialize an empty session store."""
        self.sessions: dict[str, ConversationSession] = {}

    def create_session(self, problem_id: str | None = None) -> ConversationSession:
        """Create and register a new conversational KISC session.

        Args:
            problem_id: Optional competition problem identifier.

        Returns:
            The created ``ConversationSession``.
        """
        session_id = f"kisc_sess_{uuid.uuid4().hex[:8]}"
        session = ConversationSession(
            session_id=session_id,
            created_at=int(time.time() * 1000),
            turns=[],
            feedback=FrameFeedback(),
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ConversationSession:
        """Retrieve an active session by its ID.

        Args:
            session_id: Unique session string.

        Returns:
            The matching ``ConversationSession``.

        Raises:
            KeyError: If session_id is not found.
        """
        if session_id not in self.sessions:
            raise KeyError(f"KISC session {session_id!r} not found")
        return self.sessions[session_id]

    def update_feedback(
        self, session_id: str, new_feedback: FrameFeedback
    ) -> ConversationSession:
        """Merge new human frame feedback into the cumulative feedback state.

        Args:
            session_id: Target session ID.
            new_feedback: ``FrameFeedback`` object with accepted/rejected IDs.

        Returns:
            Updated ``ConversationSession``.
        """
        session = self.get_session(session_id)

        acc = list(
            dict.fromkeys(
                session.feedback.accepted_frame_ids + new_feedback.accepted_frame_ids
            )
        )
        rej = list(
            dict.fromkeys(
                session.feedback.rejected_frame_ids + new_feedback.rejected_frame_ids
            )
        )

        session.feedback = FrameFeedback(
            accepted_frame_ids=acc, rejected_frame_ids=rej
        )
        return session

    def process_search(
        self, request: SearchRequest, engine: SearchEngine
    ) -> SearchResponse:
        """Process a search request, maintaining session turns and feedback.

        Args:
            request: Extended ``SearchRequest`` containing query and optional KISC fields.
            engine: Configured ``SearchEngine`` instance.

        Returns:
            ``SearchResponse`` with filtered results and attached session metadata.
        """
        session_id = request.session_id
        session = None

        if session_id:
            if session_id not in self.sessions:
                session = ConversationSession(
                    session_id=session_id,
                    created_at=int(time.time() * 1000),
                    turns=[],
                    feedback=FrameFeedback(),
                )
                self.sessions[session_id] = session
            else:
                session = self.sessions[session_id]

            if request.feedback:
                self.update_feedback(session_id, request.feedback)

            turn_id = f"turn_{len(session.turns) + 1:02d}"
            session.turns.append(
                ConversationTurn(
                    turn_id=turn_id, sender="user", message=request.query
                )
            )

        response = engine.search(request)

        if session:
            rejected = set(session.feedback.rejected_frame_ids)
            if rejected:
                response.results = [
                    r for r in response.results if r.frame_id not in rejected
                ]
                response.total_results = len(response.results)

            turn_id = f"turn_{len(session.turns) + 1:02d}"
            ai_msg = f"Retrieved {response.total_results} frame candidates."
            session.turns.append(
                ConversationTurn(
                    turn_id=turn_id, sender="ai", message=ai_msg
                )
            )

            response.session_id = session.session_id
            response.turn_id = turn_id
            response.ai_message = ai_msg

        return response

    def format_submission(
        self, frame_id: str, store: FrameStore
    ) -> SubmissionResult:
        """Format a target frame ID into the official BTC submission code.

        Args:
            frame_id: Target frame identifier string.
            store: Loaded ``FrameStore`` containing metadata records.

        Returns:
            ``SubmissionResult`` containing video_id, frame_idx, and submission_code.
        """
        record = store.get(frame_id)
        code = f"{record.video_id},{record.frame_idx}"
        return SubmissionResult(
            frame_id=record.frame_id,
            video_id=record.video_id,
            frame_idx=record.frame_idx,
            submission_code=code,
        )
