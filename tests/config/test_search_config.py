"""Tests for task-agnostic detached retrieval configuration."""

from hcmai.common.config import FusionConfig
from hcmai.common.schemas import RetrievalSource


def test_fusion_config_has_source_weights_not_task_weights() -> None:
    """Give every detached retrieval source one equal default RRF weight."""

    config = FusionConfig()

    assert set(config.source_weights) == set(RetrievalSource)
    assert all(weight == 1.0 for weight in config.source_weights.values())
    assert not hasattr(config, "task_weights")
