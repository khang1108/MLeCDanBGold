"""Public data-ingestion API for the HCMAI data pipeline.

This package converts a mounted AIC 2025 S1 dataset into a canonical
``frames.parquet`` file and exposes an in-memory ``FrameStore`` for
fast metadata lookup at runtime.

Public symbols
--------------
prepare_dataset:
    End-to-end offline pipeline (inventory → ingest → validate).
    Returns the path to the final ``frames.parquet`` file.
ingest_dataset:
    Build per-video Parquet shards and merge them into
    ``metadata/frames.parquet``.  Supports resume.
inventory_corpus:
    Scan the dataset and write summary statistics without ingesting.
validate_dataset:
    Validate a previously ingested ``frames.parquet`` against the
    source mappings and write audit artifacts.
FrameStore:
    In-memory index over ``frames.parquet`` that provides O(1) lookup
    by ``frame_id``, temporal neighbour queries, and filter-based ID
    enumeration for the retrieval pipeline.
"""

from hcmai.data.extract import (
    ingest_dataset,
    inventory_corpus,
    prepare_dataset,
)
from hcmai.data.loader import FrameStore
from hcmai.data.validate import validate_dataset

__all__ = [
    "FrameStore",
    "ingest_dataset",
    "inventory_corpus",
    "prepare_dataset",
    "validate_dataset",
]
