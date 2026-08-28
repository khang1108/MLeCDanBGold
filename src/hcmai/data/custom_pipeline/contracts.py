"""Run, stage, batch, and local artifact identity contracts.

``RunIdentity`` freezes everything that must match for two pipeline attempts
to share local resume state: dataset identity, the complete archive plan, the
artifact/model configuration fingerprint, and reused ASR lineage. Work windows
(offset/limit) and effective runtime resource values are operational attempt
history and are intentionally excluded from identity.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, field_validator


class RunIdentity(BaseModel):
    """Immutable identity a local pipeline run must match to resume state.

    Two runs may only share archive/batch/video state under the same
    ``version`` when every field here is identical; a changed field means the
    prior local state cannot be resumed and must start a new run.
    """

    model_config = ConfigDict(frozen=True)

    version: str
    source: str
    frame_store_id: str
    media_info_digest: str
    archive_plan_digest: str
    artifact_config_fingerprint: str
    model_revisions: Mapping[str, str]
    asr_lineage_digest: str

    @field_validator(
        "version",
        "source",
        "frame_store_id",
        "media_info_digest",
        "archive_plan_digest",
        "artifact_config_fingerprint",
        "asr_lineage_digest",
    )
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("RunIdentity fields must not be blank")
        return value

    @field_validator("model_revisions")
    @classmethod
    def _non_empty_revisions(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        if not value:
            raise ValueError("model_revisions must not be empty")
        for name, revision in value.items():
            if not name.strip() or not revision.strip():
                raise ValueError("model_revisions keys and values must not be blank")
        return dict(value)


__all__ = ["RunIdentity"]
