"""Tests for retrieval setup and query encoder configuration."""

from unittest.mock import Mock
import pytest

from hcmai.orchestration.retrieval_setup import _query_encoder
from hcmai.retrieval.embedding.pipeline import EmbeddingService


@pytest.fixture
def config():
    return Mock(backend="bge_m3")


@pytest.fixture
def index():
    idx = Mock()
    idx.metadata.embedding_dim = 512
    return idx


def test_query_encoder_uses_local_when_remote_env_is_absent(monkeypatch, config, index):
    monkeypatch.delenv("HCMAI_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("HCMAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("HCMAI_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("HCMAI_EMBEDDING_TIMEOUT_SECONDS", raising=False)

    fake_adapter = Mock()
    monkeypatch.setattr(EmbeddingService, "create_text_adapter", lambda cfg: fake_adapter)

    encoder = _query_encoder(config, index)
    assert encoder is fake_adapter


def test_query_encoder_does_not_fallback_when_remote_env_is_partial(monkeypatch, config, index):
    monkeypatch.setenv("HCMAI_EMBEDDING_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.delenv("HCMAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("HCMAI_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("HCMAI_EMBEDDING_TIMEOUT_SECONDS", raising=False)

    fake_adapter = Mock()
    monkeypatch.setattr(EmbeddingService, "create_text_adapter", lambda cfg: fake_adapter)

    try:
        _query_encoder(config, index)
    except KeyError as exc:
        assert "HCMAI_EMBEDDING_MODEL" in str(exc)
    else:
        raise AssertionError("partial remote configuration must fail explicitly")
