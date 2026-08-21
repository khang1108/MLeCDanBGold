"""Focused regression coverage for the segment-native ASR dense index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcmai.common.schemas.search import SearchFilters
from hcmai.retrieval.retriever.artifacts import sha256_file


def unit_vectors(count: int) -> np.ndarray:
    """Return deterministic normalized vectors with distinct exact scores."""

    vectors = np.eye(count, dtype=np.float32)
    return vectors


def segment_mapping() -> pd.DataFrame:
    """Return a minimal valid mapping that tests can mutate deliberately."""

    return pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": "s1",
                "video_id": "v1",
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 1000,
            },
            {
                "embedding_index": 1,
                "segment_id": "s2",
                "video_id": "v2",
                "segment_index": 0,
                "start_ms": 1000,
                "end_ms": 2000,
            },
        ]
    )


def refresh_checksums(output: Path) -> None:
    """Re-sign a deliberately tampered bundle to test semantic validation."""

    from hcmai.retrieval.retriever.segment.index import CHECKSUM_FILENAMES

    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["checksums"] = {
        filename: sha256_file(output / filename) for filename in CHECKSUM_FILENAMES
    }
    metadata_path.write_text(json.dumps(metadata))


@pytest.fixture
def tiny_segment_index():
    """Build three adjacent half-open ASR segments for filter tests."""

    pytest.importorskip("faiss")
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

    mapping = pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": "s1",
                "video_id": "v1",
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 1000,
            },
            {
                "embedding_index": 1,
                "segment_id": "s2",
                "video_id": "v1",
                "segment_index": 1,
                "start_ms": 1000,
                "end_ms": 2000,
            },
            {
                "embedding_index": 2,
                "segment_id": "s3",
                "video_id": "v1",
                "segment_index": 2,
                "start_ms": 2000,
                "end_ms": 3000,
            },
        ]
    )
    return SegmentDenseIndex.build(
        unit_vectors(3), mapping, dataset_version="v1", model_name="test-model"
    )


def test_segment_index_rejects_duplicate_segment_ids() -> None:
    """Segment identifiers, not frame identifiers, are the unique identity."""

    pytest.importorskip("faiss")
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

    mapping = pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": "s1",
                "video_id": "v1",
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 1000,
            },
            {
                "embedding_index": 1,
                "segment_id": "s1",
                "video_id": "v1",
                "segment_index": 1,
                "start_ms": 1000,
                "end_ms": 2000,
            },
        ]
    )

    with pytest.raises(ValueError, match="duplicate segment_id"):
        SegmentDenseIndex.build(
            unit_vectors(2), mapping, dataset_version="v1", model_name="m"
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("embedding_index", 0.5, "embedding_index"),
        ("segment_index", 0.5, "segment_index"),
        ("start_ms", 0.5, "start_ms"),
        ("end_ms", 1000.5, "end_ms"),
        ("segment_id", "", "segment_id"),
        ("segment_id", None, "segment_id"),
        ("video_id", " ", "video_id"),
        ("video_id", None, "video_id"),
    ],
)
def test_segment_index_rejects_malformed_identity_and_integer_fields(
    column: str, value: object, message: str
) -> None:
    """Build refuses values that would lose identity or truncate timeline data."""

    pytest.importorskip("faiss")
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

    mapping = segment_mapping()
    mapping[column] = mapping[column].astype(object)
    mapping.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        SegmentDenseIndex.build(
            unit_vectors(2), mapping, dataset_version="v1", model_name="m"
        )


def test_segment_index_rejects_video_ids_that_collide_after_string_conversion() -> None:
    """Posting keys cannot merge distinct raw identities such as ``1`` and ``"1"``."""

    pytest.importorskip("faiss")
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

    mapping = segment_mapping()
    mapping["video_id"] = [1, "1"]

    with pytest.raises(ValueError, match="video_id"):
        SegmentDenseIndex.build(
            unit_vectors(2), mapping, dataset_version="v1", model_name="m"
        )


def test_segment_index_rejects_forbidden_frame_identity() -> None:
    """Segment-native mapping cannot smuggle frame identity into the artifact."""

    pytest.importorskip("faiss")
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

    mapping = segment_mapping().assign(frame_id=["f1", "f2"])

    with pytest.raises(ValueError, match="frame_id"):
        SegmentDenseIndex.build(
            unit_vectors(2), mapping, dataset_version="v1", model_name="m"
        )


def test_segment_filter_uses_half_open_overlap(tiny_segment_index) -> None:
    """Segments touching either requested boundary do not overlap its interior."""

    positions = tiny_segment_index.filtered_positions(
        SearchFilters(video_ids=["v1"], start_time_ms=1000, end_time_ms=2000)
    )

    assert positions is not None
    assert tiny_segment_index.mapping.iloc[positions]["segment_id"].tolist() == ["s2"]


def test_segment_filtered_search_matches_brute_force(tiny_segment_index) -> None:
    """Subset retrieval ranks only overlapping ASR vectors by exact dot product."""

    query = np.asarray([[0.1, 0.8, 0.6]], dtype=np.float32)
    filters = SearchFilters(video_ids=["v1"], start_time_ms=1000, end_time_ms=3000)
    allowed = tiny_segment_index.filtered_positions(filters)
    assert allowed is not None
    brute_scores = query @ np.asarray(tiny_segment_index.vectors)[allowed].T
    ordering = np.argsort(-brute_scores[0], kind="stable")

    scores, positions = tiny_segment_index.search_filtered(query, 10, filters)

    assert positions[0].tolist() == allowed[ordering].tolist()
    np.testing.assert_allclose(scores[0], brute_scores[0][ordering])


def test_segment_global_search_matches_brute_force(tiny_segment_index) -> None:
    """Unfiltered FAISS search remains the exact global vector baseline."""

    query = np.asarray([[0.1, 0.8, 0.6]], dtype=np.float32)
    brute_scores = query @ np.asarray(tiny_segment_index.vectors).T
    expected = np.argsort(-brute_scores[0], kind="stable")

    scores, positions = tiny_segment_index.search(query, 10)

    assert positions[0].tolist() == expected.tolist()
    np.testing.assert_allclose(scores[0], brute_scores[0][expected])


def test_segment_save_load_preserves_segment_only_mapping(
    tiny_segment_index, tmp_path: Path
) -> None:
    """A persisted segment bundle retains intervals without frame identity fields."""

    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

    output = tiny_segment_index.save(tmp_path / "segments")
    loaded = SegmentDenseIndex.load(output)

    assert "frame_id" not in loaded.mapping.columns
    assert loaded.metadata.entity_kind == "segment"
    assert loaded.metadata.retrieval_source == "asr"
    assert loaded.mapping["segment_id"].tolist() == ["s1", "s2", "s3"]


def test_segment_load_rejects_checksum_tampering(
    tiny_segment_index, tmp_path: Path
) -> None:
    """Checksums prevent an altered segment vector artifact from being served."""

    from hcmai.retrieval.retriever.segment.index import (
        IndexArtifactError,
        SegmentDenseIndex,
    )

    output = tiny_segment_index.save(tmp_path / "segments")
    with (output / "vectors.npy").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(IndexArtifactError, match="checksum"):
        SegmentDenseIndex.load(output)


def test_segment_load_rejects_rechecksummed_non_flat_ip_index(
    tiny_segment_index, tmp_path: Path
) -> None:
    """Re-signing a non-flat IP index cannot bypass the exact-index contract."""

    faiss = pytest.importorskip("faiss")
    from hcmai.retrieval.retriever.segment.index import (
        IndexArtifactError,
        SegmentDenseIndex,
    )

    output = tiny_segment_index.save(tmp_path / "segments")
    replacement = faiss.IndexHNSWFlat(
        tiny_segment_index.index.d, 2, faiss.METRIC_INNER_PRODUCT
    )
    replacement.add(np.asarray(tiny_segment_index.vectors))
    faiss.write_index(replacement, str(output / "dense.index"))
    refresh_checksums(output)

    with pytest.raises(IndexArtifactError, match="IndexFlatIP"):
        SegmentDenseIndex.load(output)


def test_segment_load_rejects_rechecksummed_wrong_video_posting(tmp_path: Path) -> None:
    """Posting ranges must agree with the source video of every mapped vector."""

    pytest.importorskip("faiss")
    from hcmai.retrieval.retriever.segment.index import (
        IndexArtifactError,
        SegmentDenseIndex,
    )

    index = SegmentDenseIndex.build(
        unit_vectors(2), segment_mapping(), dataset_version="v1", model_name="m"
    )
    output = index.save(tmp_path / "segments")
    np.save(output / "posting_positions.npy", np.asarray([1, 0], dtype=np.int64))
    refresh_checksums(output)

    with pytest.raises(IndexArtifactError, match="posting.*video"):
        SegmentDenseIndex.load(output)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("start_ms", 0.5, "start_ms"),
        ("segment_id", "", "segment_id"),
        ("video_id", None, "video_id"),
        ("frame_id", "f1", "frame_id"),
    ],
)
def test_segment_load_rejects_rechecksummed_malformed_mapping(
    tiny_segment_index,
    tmp_path: Path,
    column: str,
    value: object,
    message: str,
) -> None:
    """Loader revalidates identity and integral fields after a forged checksum."""

    pytest.importorskip("faiss")
    from hcmai.retrieval.retriever.segment.index import (
        IndexArtifactError,
        SegmentDenseIndex,
    )

    output = tiny_segment_index.save(tmp_path / "segments")
    mapping_path = output / "segment_mapping.parquet"
    mapping = pd.read_parquet(mapping_path)
    if column in mapping:
        mapping[column] = mapping[column].astype(object)
    mapping.loc[0, column] = value
    mapping.to_parquet(mapping_path, index=False)
    refresh_checksums(output)

    with pytest.raises(IndexArtifactError, match=message):
        SegmentDenseIndex.load(output)


def test_zero_width_segment_filter_returns_no_results(tiny_segment_index) -> None:
    """An explicit empty time interval cannot match a non-empty segment."""

    filters = SearchFilters(start_time_ms=1000, end_time_ms=1000)

    positions = tiny_segment_index.filtered_positions(filters)
    scores, hits = tiny_segment_index.search_filtered(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), 10, filters
    )

    assert positions is not None
    assert positions.tolist() == []
    assert scores.shape == hits.shape == (1, 0)
