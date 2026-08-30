"""Offline construction adapter for the runtime-owned visual DenseIndex."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcmai.retrieval.retriever.dense.index import DenseIndex


def build_index(
    embeddings: np.ndarray,
    mapping: pd.DataFrame,
    *,
    dataset_version: str,
    model_name: str,
    index_type: str = "flat_ip",
    show_progress: bool = False,
) -> DenseIndex:
    """Build the established visual DenseIndex from precomputed vectors."""

    return DenseIndex.build(
        embeddings,
        mapping,
        dataset_version=dataset_version,
        model_name=model_name,
        index_type=index_type,
        show_progress=show_progress,
    )


__all__ = ["build_index"]
