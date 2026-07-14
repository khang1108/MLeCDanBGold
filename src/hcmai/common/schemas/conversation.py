from __future__ import annotations

from hcmai.common.schemas import NonEmptyString, ContractModel
from pydantic import Field

class ConversationTurn(ContractModel):
    """Public conversation turn for KISC problems."""
    turn_id: str = Field(description="ID for each conversation turn.")
    message: NonEmptyString

class FrameFeedback(ContractModel):
    """Public model to store feedback of human about frame, accepted or rejected."""
    accepted_frame_ids: list[NonEmptyString] = Field(description="A list of accepted frame IDs by human.",
                                                    default_factory=list)
    rejected_frame_ids: list[NonEmptyString] = Field(description="A list of rejected frame IDs by human",
                                                    default_factory=list)