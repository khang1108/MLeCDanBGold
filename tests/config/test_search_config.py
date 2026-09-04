"""Tests for task-agnostic detached retrieval and alignment configuration."""

import pytest

from hcmai.common.config import AlignmentConfig, FusionConfig, SearchConfig
from hcmai.retrieval.models import RetrievalSource


def test_fusion_config_has_source_weights_not_task_weights() -> None:
    """Give every detached retrieval source one equal default RRF weight."""

    config = FusionConfig()

    assert set(config.source_weights) == set(RetrievalSource)
    assert all(weight == 1.0 for weight in config.source_weights.values())
    assert not hasattr(config, "task_weights")


def test_alignment_config_keeps_only_full_corpus_decoder_fields() -> None:
    """Keep shortlist-era knobs out of the shared temporal baseline."""

    config = AlignmentConfig()

    assert config.lambda_gap == pytest.approx(1e-5)
    assert config.event_power == 1.0
    assert config.chunk_size == 65_536
    assert config.cluster_delta == 0.0
    assert set(config.model_dump()) == {
        "lambda_gap",
        "event_power",
        "chunk_size",
        "cluster_delta",
        "paths_per_video",
        "path_min_separation_ms",
    }


@pytest.mark.parametrize("retired_field", ["top_k", "max_videos", "rrf_k"])
def test_search_config_rejects_retired_alignment_shortlist_fields(
    retired_field: str,
) -> None:
    """Fail fast when stale YAML revives removed shortlist configuration."""

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        SearchConfig.model_validate({"alignment": {retired_field: 1}})
