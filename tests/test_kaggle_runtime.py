from __future__ import annotations

import pytest

from hcmai.common.config import TranscriptJobConfig
from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
from hcmai.llm.config import LLMServiceConfig
from kaggle.inference_server import KaggleRuntime, _models


def test_models_parser_rejects_unknown_and_empty_capabilities() -> None:
    assert _models("caption, OCR") == frozenset({"caption", "ocr"})
    with pytest.raises(ValueError, match="at least one"):
        _models("")
    with pytest.raises(ValueError, match="unsupported"):
        _models("caption,unknown")


def test_kaggle_runtime_reports_enabled_model_not_ready_before_load() -> None:
    runtime = KaggleRuntime(
        S3CorpusPreparationConfig.from_yaml("configs/preparation.s3.yaml"),
        LLMServiceConfig.from_yaml("llm/config.yaml"),
        TranscriptJobConfig.from_yaml("configs/enrichment.yaml"),
        frozenset({"asr"}),
    )

    readiness = runtime.readiness()

    assert readiness.ready is False
    assert readiness.models["asr"].enabled is True
    assert readiness.models["asr"].loaded is False
    assert readiness.models["asr"].revision is not None


def test_kaggle_runtime_rejects_worker_model_pin_mismatch() -> None:
    preparation = S3CorpusPreparationConfig.from_yaml(
        "configs/preparation.s3.yaml"
    )
    models = LLMServiceConfig.from_yaml("llm/config.yaml")
    models = models.model_copy(update={
        "caption_generation": models.caption_generation.model_copy(
            update={"revision": "f" * 40}
        )
    })

    with pytest.raises(ValueError, match="model pins differ"):
        KaggleRuntime(
            preparation,
            models,
            TranscriptJobConfig.from_yaml("configs/enrichment.yaml"),
            frozenset({"caption"}),
        )
