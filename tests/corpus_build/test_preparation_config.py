"""Production configuration tests for the S3-first corpus preparation job."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hcmai.data.corpus_build import S3CorpusPreparationConfig


SHA = "a" * 40


def _values(root: Path) -> dict[str, object]:
    work_root = (root / "run").resolve()
    return {
        "corpus_revision": "hcmai2026-videos-20260813-v1",
        "work_root": work_root,
        "stages": {
            "frame_store": True,
            "caption": True,
            "ocr": True,
            "objects": True,
            "asr": True,
            "frame_context": True,
            "visual_index": True,
            "caption_index": True,
            "ocr_index": True,
            "asr_index": True,
        },
        "models": {
            role: {"model_name": f"fixture/{role}", "revision": SHA}
            for role in (
                "caption",
                "ocr",
                "asr",
                "diarization",
                "visual_embedding",
                "text_embedding",
            )
        },
        "preprocessing": {
            "s3": {
                "bucket": "hcmai-dataset",
                "videos_prefix": "/videos/",
                "artifacts_prefix": "/artifacts/production/corpus-v1/",
                "smoke_artifacts_prefix": "/artifacts/smoke/corpus-v1/",
                "staging_root": work_root / "staging",
            },
        },
    }


def test_production_config_accepts_only_isolated_s3_inputs(tmp_path: Path) -> None:
    config = S3CorpusPreparationConfig.model_validate(_values(tmp_path))

    assert config.preprocessing.s3 is not None
    assert config.preprocessing.s3.videos_prefix == "videos"
    assert config.full_artifacts_prefix == "artifacts/production/corpus-v1"
    assert config.smoke_artifacts_prefix == "artifacts/smoke/corpus-v1"
    assert config.artifacts_root == config.work_root / "artifacts"


def test_staging_root_is_expanded_before_runtime_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist the expanded staging path instead of leaving a ``~`` literal."""

    monkeypatch.setenv("HOME", str(tmp_path))
    values = _values(tmp_path)
    preprocessing = values["preprocessing"]
    assert isinstance(preprocessing, dict)
    storage = preprocessing["s3"]
    assert isinstance(storage, dict)
    storage["staging_root"] = Path("~/run/staging")

    config = S3CorpusPreparationConfig.model_validate(values)

    assert config.preprocessing.s3 is not None
    assert config.preprocessing.s3.staging_root == (
        tmp_path / "run/staging"
    ).resolve()


def test_thunder_cache_and_execution_policy_are_validated(tmp_path: Path) -> None:
    values = _values(tmp_path)
    preprocessing = values["preprocessing"]
    assert isinstance(preprocessing, dict)
    storage = preprocessing["s3"]
    assert isinstance(storage, dict)
    work_root = values["work_root"]
    assert isinstance(work_root, Path)
    storage["cache_root"] = work_root / "source-cache"
    values["execution"] = {
        "cache_download_workers": 4,
        "minimum_free_gib_after_cache": 80,
        "overlap_frame_asr": True,
    }

    config = S3CorpusPreparationConfig.model_validate(values)

    assert config.preprocessing.s3 is not None
    assert config.preprocessing.s3.cache_root == (
        config.work_root / "source-cache"
    ).resolve()
    assert config.execution.cache_download_workers == 4
    assert config.execution.minimum_free_gib_after_cache == 80
    assert config.execution.overlap_frame_asr is True


def test_overlap_requires_a_persistent_source_cache(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["execution"] = {"overlap_frame_asr": True}

    with pytest.raises(ValidationError, match="persistent source cache"):
        S3CorpusPreparationConfig.model_validate(values)


def test_source_cache_must_stay_inside_work_root(tmp_path: Path) -> None:
    values = _values(tmp_path)
    preprocessing = values["preprocessing"]
    assert isinstance(preprocessing, dict)
    storage = preprocessing["s3"]
    assert isinstance(storage, dict)
    storage["cache_root"] = (tmp_path / "other-run/cache").resolve()

    with pytest.raises(ValidationError, match="cache_root must be inside work_root"):
        S3CorpusPreparationConfig.model_validate(values)


def test_s3_location_accepts_environment_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = S3CorpusPreparationConfig.model_validate(_values(tmp_path))
    path = tmp_path / "preparation.yaml"
    path.write_text(
        yaml.safe_dump({"preparation": baseline.model_dump(mode="json")}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HCMAI_S3_BUCKET", "verified-us-bucket")
    monkeypatch.setenv("HCMAI_S3_REGION", "us-east-2")

    loaded = S3CorpusPreparationConfig.from_yaml(path)

    assert loaded.preprocessing.s3 is not None
    assert loaded.preprocessing.s3.bucket == "verified-us-bucket"
    assert loaded.preprocessing.s3.region == "us-east-2"


def test_stage_toggles_allow_a_dependency_complete_partial_run(
    tmp_path: Path,
) -> None:
    values = _values(tmp_path)
    values["stages"] = {
        "frame_store": True,
        "caption": False,
        "ocr": False,
        "objects": False,
        "asr": False,
        "frame_context": False,
        "visual_index": True,
        "caption_index": False,
        "ocr_index": False,
        "asr_index": False,
    }

    config = S3CorpusPreparationConfig.model_validate(values)

    assert config.stages.visual_index is True
    assert config.stages.caption is False


def test_stage_toggles_reject_missing_frame_store_dependency(
    tmp_path: Path,
) -> None:
    values = _values(tmp_path)
    stages = values["stages"]
    assert isinstance(stages, dict)
    stages["frame_store"] = False

    with pytest.raises(ValidationError, match="require frame_store"):
        S3CorpusPreparationConfig.model_validate(values)


def test_production_config_requires_s3_storage(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["preprocessing"] = {}

    with pytest.raises(ValidationError, match="requires S3 storage"):
        S3CorpusPreparationConfig.model_validate(values)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("floating_corpus", "immutable corpus revision"),
        ("floating_model", "40-character"),
    ],
)
def test_production_config_rejects_unpinned_inputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    values = deepcopy(_values(tmp_path))
    preprocessing = values["preprocessing"]
    models = values["models"]
    assert isinstance(preprocessing, dict) and isinstance(models, dict)
    if mutation == "floating_corpus":
        values["corpus_revision"] = "latest"
    elif mutation == "floating_model":
        model = models["caption"]
        assert isinstance(model, dict)
        model["revision"] = "main"
    else:
        model = models["caption"]
        assert isinstance(model, dict)
        model["revision"] = "b" * 40

    with pytest.raises(ValidationError, match=message):
        S3CorpusPreparationConfig.model_validate(values)


@pytest.mark.parametrize("legacy_name", ["data", "artifacts"])
def test_production_config_rejects_repository_legacy_roots(
    tmp_path: Path,
    legacy_name: str,
) -> None:
    values = _values(tmp_path)
    project_root = Path(__file__).resolve().parents[2]
    work_root = project_root / legacy_name / "new-corpus"
    values["work_root"] = work_root
    preprocessing = values["preprocessing"]
    assert isinstance(preprocessing, dict)
    preprocessing["output_root"] = work_root / "artifacts/frame_store"
    storage = preprocessing["s3"]
    assert isinstance(storage, dict)
    storage["staging_root"] = work_root / "staging"

    with pytest.raises(ValidationError, match="legacy local"):
        S3CorpusPreparationConfig.model_validate(values)


def test_production_config_requires_separate_smoke_publication(
    tmp_path: Path,
) -> None:
    values = _values(tmp_path)
    preprocessing = values["preprocessing"]
    assert isinstance(preprocessing, dict)
    storage = preprocessing["s3"]
    assert isinstance(storage, dict)
    storage["smoke_artifacts_prefix"] = storage["artifacts_prefix"]

    with pytest.raises(ValidationError, match="smoke and full"):
        S3CorpusPreparationConfig.model_validate(values)
