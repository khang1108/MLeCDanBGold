"""Validate the hosted text request boundary used by offline context batches."""

import pytest
from pydantic import ValidationError

from llm.contracts.embeddings import TextEmbeddingRequest


def test_text_request_accepts_512_items_in_order():
    """The configured 512-caption batch must survive API validation intact."""
    texts = [f"caption {index}" for index in range(512)]
    request = TextEmbeddingRequest(model="BAAI/bge-m3", input=texts)
    assert request.input == texts


@pytest.mark.parametrize("size", [0, 513])
def test_text_request_rejects_out_of_bounds_batches(size):
    """Keep an explicit request ceiling and reject empty batches."""
    with pytest.raises(ValidationError):
        TextEmbeddingRequest(model="BAAI/bge-m3", input=["caption"] * size)
