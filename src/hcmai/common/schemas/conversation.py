from __future__ import annotations

from hcmai.common.schemas import NonEmptyString, ContractModel
from pydantic import Field

class ConversationTurn(ContractModel):
    """Public conversation turn for KISC problems."""
    turn_id: str = Field(description="ID for each conversation turn.")
    sender: str = Field(default="user", description="Sender role: 'user' or 'ai'.")
    message: NonEmptyString


class FrameFeedback(ContractModel):
    """Public model to store feedback of human about frame, accepted or rejected."""
    accepted_frame_ids: list[NonEmptyString] = Field(description="A list of accepted frame IDs by human.",
                                                    default_factory=list)
    rejected_frame_ids: list[NonEmptyString] = Field(description="A list of rejected frame IDs by human",
                                                    default_factory=list)


class ConversationSession(ContractModel):
    """Active conversational KISC session state."""
    session_id: NonEmptyString = Field(description="Unique session identifier.")
    created_at: int = Field(ge=0, description="Creation timestamp in milliseconds.")
    turns: list[ConversationTurn] = Field(default_factory=list)
    feedback: FrameFeedback = Field(default_factory=FrameFeedback)


class SubmissionResult(ContractModel):
    """Official BTC competition submission code output."""
    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    submission_code: NonEmptyString = Field(description="Official format 'video_id,frame_idx'")