"""Indexed access to offline text-enrichment artifacts."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Iterator
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from hcmai.common.schemas import FrameEnrichment, ProcessingStatus, RetrievalSource


_NULLABLE_FIELDS = (
    "caption",
    "detailed_caption",
    "ocr_text",
    "asr_text",
    "enrichment_version",
    "error_message",
)

class _TextEvidenceStore:
    """Read and index one text channel from a shared enrichment artifact."""

    source: ClassVar[RetrievalSource]
    text_field: ClassVar[str]

    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact_path = Path(artifact_path)
        table = pd.read_parquet(self.artifact_path)
        required = {"frame_id", "model_name", self.text_field}
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(
                f"{self.artifact_path} is missing columns: {', '.join(missing)}"
            )

        records = tuple(
            _materialize(row) for row in table.to_dict(orient="records")
        )
        self._by_frame_id: dict[str, FrameEnrichment] = {}
        for record in records:
            if record.frame_id in self._by_frame_id:
                raise ValueError(
                    f"Duplicate frame_id {record.frame_id!r} "
                    f"in {self.artifact_path}"
                )
            self._by_frame_id[record.frame_id] = record
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, frame_id: str) -> FrameEnrichment:
        """Return the validated enrichment row for an exact frame ID."""

        try:
            return self._by_frame_id[frame_id]
        except KeyError:
            raise KeyError(
                f"Unknown frame_id {frame_id!r} in {self.artifact_path}"
            ) from None

    def get_many(self, frame_ids: Iterable[str]) -> list[FrameEnrichment]:
        """Preserve the requested order and duplicate IDs."""

        return [self.get(frame_id) for frame_id in frame_ids]

    def get_text(self, frame_id: str) -> str | None:
        """Return usable source text, excluding failed or incomplete rows."""

        record = self.get(frame_id)
        if (
            record.status != ProcessingStatus.COMPLETED
            or record.error_message is not None
        ):
            return None
        value = getattr(record, self.text_field)
        return value if isinstance(value, str) else None

    def iter_records(self) -> Iterator[FrameEnrichment]:
        """Iterate rows in their artifact order."""

        return iter(self._records)


class CaptionStore(_TextEvidenceStore):
    """Indexed access to frame captions."""

    source = RetrievalSource.CAPTION
    text_field = "caption"


class OCRStore(_TextEvidenceStore):
    """Indexed access to text visible in frames."""

    source = RetrievalSource.OCR
    text_field = "ocr_text"


class ASRStore(_TextEvidenceStore):
    """Indexed access to speech aligned with frames."""

    source = RetrievalSource.ASR
    text_field = "asr_text"


def _is_null(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, Real) and math.isnan(float(value))


def _materialize(data: dict[Hashable, Any]) -> FrameEnrichment:
    values: dict[str, object] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Enrichment column name must be a string: {key!r}")
        values[key] = value
    objects = values.get("objects")
    to_list = getattr(objects, "tolist", None)
    if callable(to_list):
        values["objects"] = to_list()
    elif _is_null(objects):
        values["objects"] = []
    for field in _NULLABLE_FIELDS:
        if _is_null(values.get(field)):
            values[field] = None
    return FrameEnrichment.model_validate(values)
