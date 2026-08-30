"""Read transcript artifacts for offline validation and artifact builders.

Runtime timeline lookup lives in :mod:`hcmai.corpus.stores.transcript` and
returns compact runtime segments.  Offline builders still need the complete
Pydantic artifact rows, including status and provenance, so they read them
through this separate validation boundary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hcmai.common.schemas import TranscriptSegment


def load_transcript_artifact_records(
    metadata_path: str | Path,
) -> tuple[TranscriptSegment, ...]:
    """Load complete validated transcript artifact rows in stable file order.

    This preserves the existing Parquet parsing and duplicate-ID checks for
    offline publishing/index construction without exposing provenance through
    the runtime ``Corpus`` facade.
    """

    path = Path(metadata_path)
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    records: list[TranscriptSegment] = []
    for artifact_path in paths:
        table = pd.read_parquet(artifact_path).astype(object)
        rows = table.where(table.notna(), None).to_dict(orient="records")
        records.extend(TranscriptSegment.model_validate(row) for row in rows)

    identifiers = [record.segment_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate segment_id values in {path}")
    return tuple(records)
