"""Local A6000, CPU, disk, and archive-window configuration.

These contracts are engineering guardrails for the bounded local pipeline: byte
budgets, CPU/GPU scheduling ceilings, pilot model batch sizes, and the ordered
archive-plan/work-window selection. They intentionally hold no cloud
destination fields; publication is an operator-owned step outside this
package. See ``contracts.py`` for the immutable run-identity contract that
freezes lineage across attempts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hcmai.common.utils.io import read_yaml_section
from hcmai.common.utils.logging import get_logger

logger = get_logger(__name__)

_BYTES_PER_GIB = 1024**3


def _gib_to_bytes(value: float) -> int:
    """Convert a GiB value to an exact byte count for admission checks."""

    return int(value * _BYTES_PER_GIB)


class DiskBudgetConfig(BaseModel):
    """Byte-exact local disk guardrails for the active work tree.

    ``min_free_gib`` and ``max_active_gib`` are the disk admission invariant
    used before every material write: real free bytes must stay above the
    reserve and the active working set must stay below the cap.
    """

    model_config = ConfigDict(frozen=True)

    min_free_gib: float = Field(default=15.0, gt=0)
    max_active_gib: float = Field(default=30.0, gt=0)
    max_archive_download_gib: float = Field(default=20.0, gt=0)
    max_archive_uncompressed_gib: float = Field(default=25.0, gt=0)

    @model_validator(mode="after")
    def _check_reserve_fits_below_cap(self) -> "DiskBudgetConfig":
        # A reserve at or above the active cap would make every batch write
        # unconditionally inadmissible, which is never a usable profile.
        if self.min_free_gib >= self.max_active_gib and self.max_active_gib > 0:
            logger.warning(
                "disk budget reserve (%.2f GiB) is not below the active cap "
                "(%.2f GiB); every admission check will fail",
                self.min_free_gib,
                self.max_active_gib,
            )
        return self

    @property
    def min_free_bytes(self) -> int:
        """Minimum real free bytes that must remain after any planned write."""

        return _gib_to_bytes(self.min_free_gib)

    @property
    def max_active_bytes(self) -> int:
        """Maximum bytes the current active working set may occupy."""

        return _gib_to_bytes(self.max_active_gib)

    @property
    def max_archive_download_bytes(self) -> int:
        """Maximum accepted compressed size for one archive ZIP."""

        return _gib_to_bytes(self.max_archive_download_gib)

    @property
    def max_archive_uncompressed_bytes(self) -> int:
        """Maximum accepted extracted size for one archive ZIP."""

        return _gib_to_bytes(self.max_archive_uncompressed_gib)


class SchedulingConfig(BaseModel):
    """CPU/GPU worker ceilings for one measured A6000 / 6-vCPU machine.

    Only one phase (extraction, a GPU model stage, or CPU-only compaction) is
    authoritative at a time, so each phase's own worker total is validated
    against ``available_cpus`` rather than the sum across phases.
    """

    model_config = ConfigDict(frozen=True)

    available_cpus: int = Field(default=6, gt=0)
    max_videos_per_batch: int = Field(default=8, gt=0)
    extractor_processes: int = Field(default=2, gt=0)
    ffmpeg_threads_per_process: int = Field(default=2, gt=0)
    image_workers: int = Field(default=3, gt=0)
    prefetch_batches: int = Field(default=2, gt=0)
    cpu_only_threads: int = Field(default=6, gt=0)

    @model_validator(mode="after")
    def _reject_cpu_oversubscription(self) -> "SchedulingConfig":
        extraction_total = self.extractor_processes * self.ffmpeg_threads_per_process
        if extraction_total > self.available_cpus:
            raise ValueError(
                "extractor_processes * ffmpeg_threads_per_process "
                f"({extraction_total}) exceeds available_cpus "
                f"({self.available_cpus})"
            )
        if self.image_workers > self.available_cpus:
            raise ValueError(
                f"image_workers ({self.image_workers}) exceeds available_cpus "
                f"({self.available_cpus})"
            )
        if self.cpu_only_threads > self.available_cpus:
            raise ValueError(
                f"cpu_only_threads ({self.cpu_only_threads}) exceeds "
                f"available_cpus ({self.available_cpus})"
            )
        return self


class StageBatchConfig(BaseModel):
    """Pilot per-stage model batch sizes; PROPOSED until measured on A6000.

    Every value must stay at or above ``minimum`` so OOM backoff (Task 7) has
    a well-defined floor to halve toward.
    """

    model_config = ConfigDict(frozen=True)

    caption: int = Field(default=8, gt=0)
    ocr: int = Field(default=32, gt=0)
    objects: int = Field(default=32, gt=0)
    visual: int = Field(default=128, gt=0)
    context: int = Field(default=128, gt=0)
    minimum: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def _check_batches_above_minimum(self) -> "StageBatchConfig":
        for name in ("caption", "ocr", "objects", "visual", "context"):
            value = getattr(self, name)
            if value < self.minimum:
                raise ValueError(
                    f"stage batch {name!r} ({value}) is below minimum "
                    f"({self.minimum})"
                )
        return self


def _canonical_json(value: object) -> str:
    """Serialize a value in the stable representation used for digests."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _derive_archive_id(url: str) -> str:
    """Derive a stable, filesystem-safe archive ID from a ZIP URL's filename."""

    path = urlparse(url).path
    stem = path.rsplit("/", maxsplit=1)[-1]
    if stem.lower().endswith(".zip"):
        stem = stem[: -len(".zip")]
    if not stem:
        raise ValueError(f"cannot derive an archive_id from url: {url}")
    return stem


def _compute_plan_digest(entries: Sequence["ArchivePlanEntry"]) -> str:
    """Compute a deterministic SHA-256 digest over the ordered archive plan."""

    payload = [
        {"position": entry.position, "archive_id": entry.archive_id, "url": entry.url}
        for entry in entries
    ]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class ArchivePlanEntry(BaseModel):
    """One organizer ZIP position in the immutable ordered archive plan."""

    model_config = ConfigDict(frozen=True)

    position: int = Field(ge=0)
    archive_id: str
    url: str

    @field_validator("archive_id")
    @classmethod
    def _non_blank_archive_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("archive_id must not be blank")
        return value

    @field_validator("url")
    @classmethod
    def _https_zip_url(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError(f"archive url must use https: {value}")
        if not value.lower().endswith(".zip"):
            raise ValueError(f"archive url must reference a .zip file: {value}")
        return value


class ArchivePlan(BaseModel):
    """The complete, ordered, digested organizer ZIP plan from the runbook.

    The digest is part of the frozen run identity: two runs may only share
    state if they agree on the exact ordered set of archive URLs.
    """

    model_config = ConfigDict(frozen=True)

    entries: tuple[ArchivePlanEntry, ...]
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _check_plan_is_ordered_and_unique(self) -> "ArchivePlan":
        if not self.entries:
            raise ValueError("archive plan must not be empty")
        positions = [entry.position for entry in self.entries]
        if positions != list(range(len(self.entries))):
            raise ValueError("archive plan positions must be contiguous from zero")
        archive_ids = [entry.archive_id for entry in self.entries]
        if len(set(archive_ids)) != len(archive_ids):
            raise ValueError("archive plan archive_id values must be unique")
        urls = [entry.url for entry in self.entries]
        if len(set(urls)) != len(urls):
            raise ValueError("archive plan urls must be unique")
        expected_digest = _compute_plan_digest(self.entries)
        if self.digest != expected_digest:
            raise ValueError("archive plan digest does not match its entries")
        return self

    @classmethod
    def from_urls(cls, urls: Sequence[str]) -> "ArchivePlan":
        """Build an ordered, digested plan from the runbook's URL array."""

        entries = tuple(
            ArchivePlanEntry(position=index, archive_id=_derive_archive_id(url), url=url)
            for index, url in enumerate(urls)
        )
        digest = _compute_plan_digest(entries)
        logger.info(
            "built archive plan: %d archives, digest=%s", len(entries), digest[:12]
        )
        return cls(entries=entries, digest=digest)


class ArchiveWorkWindow(BaseModel):
    """One operational zero-based archive-position slice of the archive plan.

    Work windows are attempt history, not artifact identity: replaying an
    adjacent or overlapping window against the same plan is a normal resume
    operation, not a new run.
    """

    model_config = ConfigDict(frozen=True)

    offset: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, gt=0)

    def select(self, plan: ArchivePlan) -> tuple[ArchivePlanEntry, ...]:
        """Return the non-empty ordered slice of ``plan`` for this window.

        Raises:
            ValueError: If ``offset`` is at or beyond the end of the plan,
                which would select zero archives.
        """

        total = len(plan.entries)
        if self.offset >= total:
            raise ValueError(
                f"offset {self.offset} is out of range for a {total}-archive plan"
            )
        end = total if self.limit is None else min(total, self.offset + self.limit)
        window = plan.entries[self.offset : end]
        logger.info(
            "selected archive work window offset=%d limit=%s -> positions %d..%d",
            self.offset,
            self.limit,
            window[0].position,
            window[-1].position,
        )
        return window


class CustomPipelineConfig(BaseModel):
    """Local roots, device, and bundled disk/scheduling/batch settings.

    Loaded from the ``custom_pipeline`` section of ``configs/prepare.yaml``.
    This section is additive: it does not alter any BTC-keyframe default
    elsewhere in the file, and it holds no cloud destination field.
    """

    model_config = ConfigDict(frozen=True)

    run_root: str
    artifacts_root: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    disk: DiskBudgetConfig = Field(default_factory=DiskBudgetConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    stage_batches: StageBatchConfig = Field(default_factory=StageBatchConfig)

    @field_validator("run_root", "artifacts_root", "device", "dtype")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("custom_pipeline fields must not be blank")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CustomPipelineConfig":
        """Load the local A6000 profile from the shared preparation config."""

        raw = read_yaml_section(path, "custom_pipeline")
        disk = DiskBudgetConfig(**raw.get("disk", {}))
        scheduling = SchedulingConfig(**raw.get("scheduling", {}))
        stage_batches = StageBatchConfig(**raw.get("stage_batches", {}))
        missing = sorted({"run_root", "artifacts_root"} - set(raw))
        if missing:
            raise ValueError(f"custom_pipeline config missing: {', '.join(missing)}")
        config = cls(
            run_root=raw["run_root"],
            artifacts_root=raw["artifacts_root"],
            device=raw.get("device", "cuda"),
            dtype=raw.get("dtype", "bfloat16"),
            disk=disk,
            scheduling=scheduling,
            stage_batches=stage_batches,
        )
        logger.info(
            "loaded custom_pipeline config: run_root=%s artifacts_root=%s device=%s",
            config.run_root,
            config.artifacts_root,
            config.device,
        )
        return config


__all__ = [
    "ArchivePlan",
    "ArchivePlanEntry",
    "ArchiveWorkWindow",
    "CustomPipelineConfig",
    "DiskBudgetConfig",
    "SchedulingConfig",
    "StageBatchConfig",
]
