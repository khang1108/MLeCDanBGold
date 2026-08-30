"""Tests for the local A6000 pipeline configuration and identity contracts.

Covers byte-exact disk budgets, CPU oversubscription rejection, stage batch
floors, ordered/digested archive plans, work-window selection, and proof that
no cloud destination field exists in serialized config or run identity.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from offline.ingestion.custom_pipeline.config import (
    ArchivePlan,
    ArchiveWorkWindow,
    CustomPipelineConfig,
    DiskBudgetConfig,
    SchedulingConfig,
    StageBatchConfig,
)
from offline.ingestion.custom_pipeline.contracts import RunIdentity


# ---------------------------------------------------------------------------
# DiskBudgetConfig
# ---------------------------------------------------------------------------


def test_disk_budget_converts_gib_to_exact_bytes() -> None:
    budget = DiskBudgetConfig()
    assert budget.min_free_bytes == 15 * 1024**3
    assert budget.max_active_bytes == 30 * 1024**3
    assert budget.max_archive_download_bytes == 20 * 1024**3
    assert budget.max_archive_uncompressed_bytes == 25 * 1024**3


@pytest.mark.parametrize(
    "field",
    ["min_free_gib", "max_active_gib", "max_archive_download_gib", "max_archive_uncompressed_gib"],
)
def test_disk_budget_rejects_non_positive_values(field: str) -> None:
    with pytest.raises(ValidationError):
        DiskBudgetConfig(**{field: 0})
    with pytest.raises(ValidationError):
        DiskBudgetConfig(**{field: -1})


# ---------------------------------------------------------------------------
# SchedulingConfig
# ---------------------------------------------------------------------------


def test_scheduling_config_default_fits_six_cpus() -> None:
    scheduling = SchedulingConfig()
    assert scheduling.available_cpus == 6
    assert scheduling.extractor_processes * scheduling.ffmpeg_threads_per_process <= 6


def test_scheduling_config_rejects_extractor_oversubscription() -> None:
    with pytest.raises(ValidationError, match="exceeds available_cpus"):
        SchedulingConfig(extractor_processes=4, ffmpeg_threads_per_process=2, available_cpus=6)


def test_scheduling_config_rejects_image_worker_oversubscription() -> None:
    with pytest.raises(ValidationError, match="exceeds available_cpus"):
        SchedulingConfig(image_workers=10, available_cpus=6)


def test_scheduling_config_rejects_cpu_only_thread_oversubscription() -> None:
    with pytest.raises(ValidationError, match="exceeds available_cpus"):
        SchedulingConfig(cpu_only_threads=10, available_cpus=6)


@pytest.mark.parametrize(
    "field",
    ["max_videos_per_batch", "extractor_processes", "ffmpeg_threads_per_process", "image_workers", "prefetch_batches", "cpu_only_threads"],
)
def test_scheduling_config_rejects_non_positive_worker_values(field: str) -> None:
    with pytest.raises(ValidationError):
        SchedulingConfig(**{field: 0})


# ---------------------------------------------------------------------------
# StageBatchConfig
# ---------------------------------------------------------------------------


def test_stage_batch_config_defaults_are_pilot_values() -> None:
    batches = StageBatchConfig()
    assert (batches.caption, batches.ocr, batches.objects, batches.visual, batches.context) == (
        8,
        32,
        32,
        128,
        128,
    )


@pytest.mark.parametrize("field", ["caption", "ocr", "objects", "visual", "context"])
def test_stage_batch_config_rejects_non_positive_batches(field: str) -> None:
    with pytest.raises(ValidationError):
        StageBatchConfig(**{field: 0})


def test_stage_batch_config_rejects_batch_below_minimum() -> None:
    with pytest.raises(ValidationError, match="below minimum"):
        StageBatchConfig(caption=2, minimum=4)


# ---------------------------------------------------------------------------
# ArchivePlan
# ---------------------------------------------------------------------------

_URLS = [
    "https://example.org/archives/L01.zip",
    "https://example.org/archives/L02.zip",
    "https://example.org/archives/L03.zip",
    "https://example.org/archives/L04.zip",
    "https://example.org/archives/L05.zip",
]


def test_archive_plan_from_urls_is_ordered_and_digested() -> None:
    plan = ArchivePlan.from_urls(_URLS)
    assert [entry.position for entry in plan.entries] == [0, 1, 2, 3, 4]
    assert [entry.archive_id for entry in plan.entries] == ["L01", "L02", "L03", "L04", "L05"]
    assert len(plan.digest) == 64


def test_archive_plan_digest_is_deterministic_for_same_urls() -> None:
    first = ArchivePlan.from_urls(_URLS)
    second = ArchivePlan.from_urls(list(_URLS))
    assert first.digest == second.digest


def test_archive_plan_digest_changes_when_urls_change() -> None:
    first = ArchivePlan.from_urls(_URLS)
    second = ArchivePlan.from_urls([*_URLS, "https://example.org/archives/L06.zip"])
    assert first.digest != second.digest


def test_archive_plan_rejects_non_https_url() -> None:
    with pytest.raises(ValidationError, match="https"):
        ArchivePlan.from_urls(["http://example.org/archives/L01.zip"])


def test_archive_plan_rejects_non_zip_url() -> None:
    with pytest.raises(ValidationError, match=r"\.zip"):
        ArchivePlan.from_urls(["https://example.org/archives/L01.tar"])


def test_archive_plan_rejects_duplicate_urls() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ArchivePlan.from_urls([_URLS[0], _URLS[0]])


def test_archive_plan_rejects_empty_url_list() -> None:
    with pytest.raises(ValidationError):
        ArchivePlan.from_urls([])


# ---------------------------------------------------------------------------
# ArchiveWorkWindow
# ---------------------------------------------------------------------------


def test_default_window_selects_the_whole_plan() -> None:
    plan = ArchivePlan.from_urls(_URLS)
    window = ArchiveWorkWindow()
    selected = window.select(plan)
    assert [entry.position for entry in selected] == [0, 1, 2, 3, 4]


def test_offset_and_limit_select_the_expected_positions() -> None:
    plan = ArchivePlan.from_urls(_URLS)
    window = ArchiveWorkWindow(offset=2, limit=3)
    selected = window.select(plan)
    assert [entry.position for entry in selected] == [2, 3, 4]


def test_omitted_limit_selects_from_offset_to_end() -> None:
    plan = ArchivePlan.from_urls(_URLS)
    window = ArchiveWorkWindow(offset=3)
    selected = window.select(plan)
    assert [entry.position for entry in selected] == [3, 4]


def test_out_of_range_offset_is_rejected() -> None:
    plan = ArchivePlan.from_urls(_URLS)
    window = ArchiveWorkWindow(offset=5)
    with pytest.raises(ValueError, match="out of range"):
        window.select(plan)


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ArchiveWorkWindow(limit=0)


# ---------------------------------------------------------------------------
# RunIdentity — no cloud destination fields anywhere
# ---------------------------------------------------------------------------

_FORBIDDEN_TOKENS = ("s3", "bucket", "cloud", "upload", "sync", "remote")


def _make_run_identity() -> RunIdentity:
    return RunIdentity(
        version="dataset_v1",
        source="custom_raw_video_1fps",
        frame_store_id="dataset_1",
        media_info_digest="a" * 64,
        archive_plan_digest="b" * 64,
        artifact_config_fingerprint="c" * 64,
        model_revisions={"caption": "rev1", "ocr": "rev2"},
        asr_lineage_digest="d" * 64,
    )


def test_run_identity_serialization_has_no_cloud_destination_fields() -> None:
    identity = _make_run_identity()
    serialized = identity.model_dump()
    for key in serialized:
        assert not any(token in key.lower() for token in _FORBIDDEN_TOKENS), key


def test_disk_budget_serialization_has_no_cloud_destination_fields() -> None:
    serialized = DiskBudgetConfig().model_dump()
    for key in serialized:
        assert not any(token in key.lower() for token in _FORBIDDEN_TOKENS), key


def test_run_identity_rejects_blank_fields() -> None:
    with pytest.raises(ValidationError):
        RunIdentity(
            version="",
            source="custom_raw_video_1fps",
            frame_store_id="dataset_1",
            media_info_digest="a" * 64,
            archive_plan_digest="b" * 64,
            artifact_config_fingerprint="c" * 64,
            model_revisions={"caption": "rev1"},
            asr_lineage_digest="d" * 64,
        )


def test_run_identity_rejects_empty_model_revisions() -> None:
    with pytest.raises(ValidationError, match="model_revisions"):
        RunIdentity(
            version="dataset_v1",
            source="custom_raw_video_1fps",
            frame_store_id="dataset_1",
            media_info_digest="a" * 64,
            archive_plan_digest="b" * 64,
            artifact_config_fingerprint="c" * 64,
            model_revisions={},
            asr_lineage_digest="d" * 64,
        )


# ---------------------------------------------------------------------------
# CustomPipelineConfig.from_yaml
# ---------------------------------------------------------------------------


def test_custom_pipeline_config_loads_from_prepare_yaml() -> None:
    config = CustomPipelineConfig.from_yaml("configs/prepare.yaml")
    assert config.run_root
    assert config.artifacts_root
    assert config.disk.min_free_gib == 15
    assert config.scheduling.available_cpus == 6
    assert config.stage_batches.caption == 8

