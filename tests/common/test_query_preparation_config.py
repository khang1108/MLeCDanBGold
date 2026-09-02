"""Tests for query-preparation runtime configuration."""

import pytest
from hcmai.common.config import QueryPreparationConfig
from pydantic import ValidationError


def test_query_preparation_defaults_are_frozen_product_defaults() -> None:
    """Keep the public candidate count and model identity reproducible."""

    config = QueryPreparationConfig()

    assert config.model_name == "Qwen/Qwen3-4B"
    assert config.model_revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert config.prompt_version == "query-prep-v1"
    assert config.candidate_count == 5
    assert config.cache_ttl_seconds == 3600
    assert config.cache_max_entries == 2048


def test_candidate_count_cannot_drift_from_public_contract() -> None:
    """Reject candidate counts that the public response cannot represent."""

    with pytest.raises(ValidationError):
        QueryPreparationConfig(candidate_count=4)


def test_model_revision_must_be_an_immutable_sha() -> None:
    """Reject mutable model revisions such as ``main``."""

    with pytest.raises(ValidationError):
        QueryPreparationConfig(model_revision="main")