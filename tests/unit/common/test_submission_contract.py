"""Tests for the task-independent competition submission contract."""

from pydantic import ValidationError
import pytest

from hcmai.api.contracts import SubmissionResult


def test_submission_contract_preserves_competition_identity() -> None:
    """Reject a submission code that rewrites the canonical BTC coordinate."""

    valid = SubmissionResult(
        frame_id="f1",
        video_id="L21_V001",
        frame_idx=10,
        submission_code="L21_V001,10",
    )

    with pytest.raises(ValidationError, match="submission_code"):
        SubmissionResult.model_validate(
            {**valid.model_dump(), "submission_code": "L21_V001,11"}
        )
