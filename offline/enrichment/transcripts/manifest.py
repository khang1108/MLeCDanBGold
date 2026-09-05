"""Transcript manifest tracking per-video completion for resumable runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from hcmai.common.config import ASRConfig, DiarizationConfig
from offline.contracts import ContractModel, NonEmptyString
from offline.enrichment.transcripts.models import TranscriptSegment
from hcmai.common.utils.io import read_json


class SourceFingerprint(ContractModel):
    """Content identity for one source video."""

    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TranscriptManifest(ContractModel):
    """Resume identity and provenance for segment-native transcript evidence."""

    video_id: NonEmptyString
    source: SourceFingerprint
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asr_model: NonEmptyString
    asr_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    diarization_enabled: bool
    diarization_model: NonEmptyString | None = None
    diarization_revision: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{40}$"
    )
    schema_version: NonEmptyString
    pipeline_version: NonEmptyString
    source_of_truth: Literal["transcript segments"] = "transcript segments"
    frame_alignment: Literal["derived compatibility view"] = (
        "derived compatibility view"
    )
    context_dependency: Literal["none"] = "none"
    segment_count: int = Field(ge=0)
    status: Literal["completed", "failed"]
    failure_category: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.diarization_enabled != (self.diarization_model is not None):
            raise ValueError("diarization model must match enabled state")
        if self.diarization_enabled != (self.diarization_revision is not None):
            raise ValueError("diarization revision must match enabled state")
        if self.status == "completed" and self.failure_category is not None:
            raise ValueError("completed manifests cannot contain failures")
        if self.status == "failed" and self.failure_category is None:
            raise ValueError("failed manifests require a failure category")
        return self

    def resume_identity(self) -> tuple[object, ...]:
        """Return every field whose mismatch must invalidate reuse."""

        return (
            self.video_id,
            self.source,
            self.config_sha256,
            self.asr_model,
            self.asr_revision,
            self.diarization_enabled,
            self.diarization_model,
            self.diarization_revision,
            self.schema_version,
            self.pipeline_version,
        )


def fingerprint_source(path: Path, *, chunk_size: int = 1_048_576) -> SourceFingerprint:
    """Hash a source video without relying on mutable path or mtime metadata."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return SourceFingerprint(size_bytes=path.stat().st_size, sha256=digest.hexdigest())


def configuration_hash(
    asr: ASRConfig,
    diarization: DiarizationConfig | None,
    *,
    schema_version: str,
    pipeline_version: str,
) -> str:
    """Hash all behavior-affecting transcript settings canonically."""

    payload = {
        "asr": asr.model_dump(mode="json"),
        "diarization": (
            diarization.model_dump(mode="json") if diarization is not None else None
        ),
        "schema_version": schema_version,
        "pipeline_version": pipeline_version,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_manifest(
    video: Path,
    video_id: str,
    asr_config: ASRConfig,
    diarization_config: DiarizationConfig | None,
    *,
    asr_revision: str,
    diarization_revision: str | None,
    schema_version: str,
    pipeline_version: str,
) -> TranscriptManifest:
    """Build a completed manifest identity before segment count is known."""

    return TranscriptManifest(
        video_id=video_id,
        source=fingerprint_source(video),
        config_sha256=configuration_hash(
            asr_config,
            diarization_config,
            schema_version=schema_version,
            pipeline_version=pipeline_version,
        ),
        asr_model=asr_config.model_name,
        asr_revision=asr_revision,
        diarization_enabled=diarization_config is not None,
        diarization_model=(
            diarization_config.model_name if diarization_config is not None else None
        ),
        diarization_revision=diarization_revision,
        schema_version=schema_version,
        pipeline_version=pipeline_version,
        segment_count=0,
        status="completed",
    )


def load_manifest(path: Path) -> TranscriptManifest:
    """Load and validate one transcript manifest."""

    return TranscriptManifest.model_validate(read_json(path))


def reusable_transcript(
    output: Path,
    manifest_path: Path,
    expected: TranscriptManifest,
    records: list[TranscriptSegment],
) -> bool:
    """Accept reuse only for a complete matching and fully validated pair."""

    if not output.is_file() or not manifest_path.is_file():
        return False
    try:
        actual = load_manifest(manifest_path)
    except Exception:
        return False
    return (
        actual.status == "completed"
        and actual.resume_identity() == expected.resume_identity()
        and actual.segment_count == len(records)
        and all(record.video_id == expected.video_id for record in records)
    )


def failure_manifest(
    expected: TranscriptManifest, error: Exception
) -> TranscriptManifest:
    """Record only a bounded exception category, never raw provider output."""

    return expected.model_copy(update={
        "status": "failed",
        "segment_count": 0,
        "failure_category": type(error).__name__[:100],
    })
