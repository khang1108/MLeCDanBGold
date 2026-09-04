"""Load and score fielded frame-native BM25 artifacts read-only.

Original Vietnamese events score title, caption, OCR, and ASR. Results are
always reordered to canonical visual-index positions.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from hcmai.common.config import BM25FieldWeights
from hcmai.retrieval.evidence.components import TemporalScoreBundle, TemporalScoreComponent
from scipy import sparse

FIELDS = ("title", "caption", "ocr", "asr")
IDENTITY_COLUMNS = ("frame_id", "video_id", "frame_idx", "timestamp_ms")
SCHEMA_VERSION = "frame-bm25-v1"
TOKENIZER_VERSION = "nfkc-alnum-v1"
K1 = 1.5
B = 0.75


class BM25ArtifactError(RuntimeError):
    """A BM25 artifact is incomplete or conflicts with canonical identity."""


class BM25TemporalScorer:
    """Score original Vietnamese events over four frame-native BM25 fields."""

    def __init__(
        self,
        matrices: dict[str, sparse.csr_matrix],
        vocabularies: dict[str, dict[str, int]],
        reorder: np.ndarray,
        field_weights: BM25FieldWeights,
    ) -> None:
        """Retain validated sparse matrices and canonical reorder positions."""

        self._matrices = matrices
        self._vocabularies = vocabularies
        self._reorder = reorder
        self._field_weights = field_weights

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        canonical_mapping: pd.DataFrame,
        field_weights: BM25FieldWeights,
    ) -> BM25TemporalScorer:
        """Load one complete artifact and verify all canonical identity columns."""

        root = Path(artifact_dir)

        required = [root / "frame_mapping.parquet", root / "metadata.json"] + [
            root / f"{field}_{suffix}"
            for field in FIELDS
            for suffix in ("vocab.json", "weights.npz")

        ]

        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise BM25ArtifactError(f"incomplete BM25 artifact: missing {', '.join(missing)}")

        artifact_mapping = pd.read_parquet(root / "frame_mapping.parquet")
        _validate_identity(artifact_mapping, canonical_mapping)
        artifact_positions = {
            str(frame_id): position
            for position, frame_id in enumerate(artifact_mapping["frame_id"])
        }
        reorder = np.asarray(
            [artifact_positions[str(frame_id)] for frame_id in canonical_mapping["frame_id"]],
            dtype=np.int64,
        )
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("document_count") != len(artifact_mapping):
            raise BM25ArtifactError("BM25 metadata document count is inconsistent")
        expected_metadata = {
            "schema_version": SCHEMA_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": K1,
            "b": B,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise BM25ArtifactError("BM25 metadata format is incompatible")

        matrices: dict[str, sparse.csr_matrix] = {}
        vocabularies: dict[str, dict[str, int]] = {}
        for field in FIELDS:
            vocabulary = json.loads((root / f"{field}_vocab.json").read_text(encoding="utf-8"))
            matrix = sparse.load_npz(root / f"{field}_weights.npz").tocsr()
            if matrix.shape != (len(artifact_mapping), len(vocabulary)):
                raise BM25ArtifactError(f"{field} BM25 matrix shape is inconsistent")
            vocabularies[field] = {
                str(token): int(position) for token, position in vocabulary.items()
            }
            matrices[field] = matrix
        return cls(matrices, vocabularies, reorder, field_weights)

    def score_components(
        self,
        original_events: Sequence[str],
        caption_events: Sequence[str],
    ) -> TemporalScoreBundle:
        """Score each BM25 field independently without field weighting."""

        if len(original_events) != len(caption_events) or not original_events:
            raise ValueError("original and caption events must have equal non-zero lengths")

        rows = {
            "bm25_title": np.zeros((len(original_events), len(self._reorder)), dtype=np.float32),
            "bm25_caption": np.zeros((len(original_events), len(self._reorder)), dtype=np.float32),
            "bm25_ocr": np.zeros((len(original_events), len(self._reorder)), dtype=np.float32),
            "bm25_asr": np.zeros((len(original_events), len(self._reorder)), dtype=np.float32),
        }
        for event_index, (original, caption) in enumerate(
            zip(original_events, caption_events, strict=True)
        ):
            original_tokens = _tokenize(original)
            rows["bm25_title"][event_index] = self._score_field("title", original_tokens)[self._reorder]
            rows["bm25_ocr"][event_index] = self._score_field("ocr", original_tokens)[self._reorder]
            rows["bm25_asr"][event_index] = self._score_field("asr", original_tokens)[self._reorder]
            rows["bm25_caption"][event_index] = self._score_field(
                "caption", _tokenize(caption)
            )[self._reorder]

        return TemporalScoreBundle(
            {name: TemporalScoreComponent(name, scores) for name, scores in rows.items()}
        )

    def score_events(
        self,
        original_events: Sequence[str],
        caption_events: Sequence[str],
    ) -> np.ndarray:
        """Score Vietnamese fields, with the caption query routed explicitly."""

        bundle = self.score_components(original_events, caption_events)
        return (
            self._field_weights.title_weight * bundle.components["bm25_title"].raw_scores
            + self._field_weights.caption_weight * bundle.components["bm25_caption"].raw_scores
            + self._field_weights.ocr_weight * bundle.components["bm25_ocr"].raw_scores
            + self._field_weights.asr_weight * bundle.components["bm25_asr"].raw_scores
        ).astype(np.float32, copy=False)

    def _score_field(self, field: str, tokens: Sequence[str]) -> np.ndarray:
        """Sum precomputed sparse columns for query terms present in one field."""

        vocabulary = self._vocabularies[field]
        matrix = cast(Any, self._matrices[field])
        columns = [vocabulary[token] for token in tokens if token in vocabulary]
        if not columns:
            return np.zeros(int(matrix.shape[0]), dtype=np.float32)
        values = np.asarray(matrix[:, columns].sum(axis=1)).reshape(-1)
        return values.astype(np.float32, copy=False)


def _validate_identity(artifact: pd.DataFrame, canonical: pd.DataFrame) -> None:
    """Verify artifact and visual mappings represent the same canonical frames."""

    for name, table in (("artifact", artifact), ("canonical", canonical)):
        missing = set(IDENTITY_COLUMNS) - set(table.columns)
        if missing:
            raise BM25ArtifactError(f"{name} mapping lacks identity columns: {sorted(missing)}")
        if table["frame_id"].duplicated().any():
            raise BM25ArtifactError(f"{name} mapping contains duplicate frame identity")
    artifact_by_id = artifact.set_index(artifact["frame_id"].astype(str))
    canonical_by_id = canonical.set_index(canonical["frame_id"].astype(str))
    if set(artifact_by_id.index) != set(canonical_by_id.index):
        raise BM25ArtifactError("BM25 frame identity set conflicts with canonical mapping")
    for frame_id in canonical_by_id.index:
        left = artifact_by_id.loc[frame_id]
        right = canonical_by_id.loc[frame_id]
        if (
            str(left["video_id"]) != str(right["video_id"])
            or int(left["frame_idx"]) != int(right["frame_idx"])
            or int(left["timestamp_ms"]) != int(right["timestamp_ms"])
        ):
            raise BM25ArtifactError("BM25 identity metadata conflicts with canonical mapping")


def _tokenize(text: str) -> list[str]:
    """Apply the artifact tokenizer contract to one runtime query."""

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(character if character.isalnum() else " " for character in normalized).split()
