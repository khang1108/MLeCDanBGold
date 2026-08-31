"""Tests for shared inference and encoder contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hcmai.common.config import EncoderConfig
from thundercompute.contracts import TextEmbeddingRequest


def test_text_embedding_contract_uses_shared_text_source_name() -> None:
    """Accept only the model gateway's shared text embedding family."""

    request = TextEmbeddingRequest(source="text", texts=["red bus"])

    assert request.source == "text"
    with pytest.raises(ValidationError):
        TextEmbeddingRequest(source="caption", texts=["red bus"])


def test_text_embedding_contract_defers_batch_ceiling_to_the_service() -> None:
    """Allow deployments to raise the model-specific request ceiling."""

    request = TextEmbeddingRequest(texts=["red bus"] * 128)

    assert len(request.texts) == 128


@pytest.mark.parametrize("batch_size", [0, -1])
def test_encoder_config_rejects_nonpositive_batch_size(batch_size: int) -> None:
    """Keep every range-based encoder caller on a positive batch size."""

    with pytest.raises(ValidationError, match="greater than 0"):
        EncoderConfig(batch_size=batch_size)
