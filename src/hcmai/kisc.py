"""State and protocol for conversational known-item search."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from hcmai.common.schemas import (
    ConversationSession,
    ConversationTurn,
    FrameFeedback,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SubmissionResult,
)
from hcmai.data import FrameStore
from hcmai.search import SearchEngine


def _now_ms() -> int:
    """Return the current Unix timestamp in milliseconds."""
    return int(time.time() * 1_000)


class KiscSessionManager:
    """Manage explicit in-memory KISC sessions and feedback."""

    def __init__(self) -> None:
        self.sessions: dict[str, ConversationSession] = {}

    def create_session(
        self,
        problem_id: str | None = None,
    ) -> ConversationSession:
        """Create and register a session for an optional problem.

        Args:
            problem_id (str): Unique ID of the problem

        Returns:
            A conversation session
        """
        session = ConversationSession(
            session_id=f"kisc_sess_{uuid.uuid4().hex[:8]}",
            created_at=_now_ms(),
            problem_id=problem_id,
        )
        self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> ConversationSession:
        """Return an existing session; KISC never creates one implicitly.

        Args:
            session_id (str): Unique ID of the session

        Returns:
            A conversation session of session id.

        Raises:
            KeyError: If not found a conversation with the given session_id.
        """
        try:
            return self.sessions[session_id]
        except KeyError as error:
            raise KeyError(f"KISC session {session_id!r} not found") from error

    def update_feedback(
        self,
        session_id: str,
        new_feedback: FrameFeedback,
    ) -> ConversationSession:
        """Merge feedback using the latest decision for each frame."""
        session = self.get_session(session_id)
        accepted = list(
            dict.fromkeys(
                session.feedback.accepted_frame_ids
                + new_feedback.accepted_frame_ids
            )
        )
        rejected = list(
            dict.fromkeys(
                session.feedback.rejected_frame_ids
                + new_feedback.rejected_frame_ids
            )
        )
        new_accepted = set(new_feedback.accepted_frame_ids)
        new_rejected = set(new_feedback.rejected_frame_ids)
        accepted = [item for item in accepted if item not in new_rejected]
        rejected = [item for item in rejected if item not in new_accepted]
        session.feedback = FrameFeedback(
            accepted_frame_ids=accepted,
            rejected_frame_ids=rejected,
        )
        return session

    def _append_turn(
        self,
        session: ConversationSession,
        sender: Literal["user", "ai"],
        message: str,
        reply_to: str | None = None,
    ) -> ConversationTurn:
        """Append one server-ordered session turn."""
        turn = ConversationTurn(
            turn_id=f"turn_{len(session.turns) + 1:04d}",
            sender=sender,
            message=message,
            created_at=_now_ms(),
            reply_to_turn_id=reply_to,
        )
        session.turns.append(turn)
        return turn

    @staticmethod
    def _apply_feedback(
        results: list[SearchResult],
        feedback: FrameFeedback,
    ) -> list[SearchResult]:
        """Remove rejected results, promote accepted ones, and reset ranks."""
        rejected = set(feedback.rejected_frame_ids)
        accepted_order = {
            frame_id: index
            for index, frame_id in enumerate(feedback.accepted_frame_ids)
        }
        kept = [
            result for result in results
            if result.frame_id not in rejected
        ]
        kept.sort(
            key=lambda result: (
                result.frame_id not in accepted_order,
                accepted_order.get(result.frame_id, result.rank),
            )
        )
        return [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(kept, start=1)
        ]

    def process_search(
        self,
        request: SearchRequest,
        engine: SearchEngine,
    ) -> SearchResponse:
        """Execute stateless search or one turn in an existing session."""
        if request.session_id is None:
            return engine.search(request)

        session = self.get_session(request.session_id)
        if request.feedback is not None:
            session = self.update_feedback(
                session.session_id,
                request.feedback,
            )
        user_turn = self._append_turn(session, "user", request.query)
        response = engine.search(request)
        results = self._apply_feedback(response.results, session.feedback)
        ai_message = f"Retrieved {len(results)} frame candidates."
        ai_turn = self._append_turn(
            session,
            "ai",
            ai_message,
            reply_to=user_turn.turn_id,
        )
        payload = response.model_dump()
        payload.update(
            total_results=len(results),
            results=results,
            session_id=session.session_id,
            turn_id=user_turn.turn_id,
            assistant_turn_id=ai_turn.turn_id,
            ai_message=ai_message,
        )
        return SearchResponse.model_validate(payload)

    def format_submission(
        self,
        frame_id: str,
        store: FrameStore,
    ) -> SubmissionResult:
        """Resolve one frame and format the official submission code."""
        record = store.get(frame_id)
        return SubmissionResult(
            frame_id=record.frame_id,
            video_id=record.video_id,
            frame_idx=record.frame_idx,
            submission_code=f"{record.video_id},{record.frame_idx}",
        )
