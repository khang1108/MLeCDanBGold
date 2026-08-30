"""Runtime retrieval binding for a loaded frame-native Context index.

Offline corpus construction, embedding, and artifact publication live in
``offline.indexes``. This module only binds an existing DenseIndex to query
encoding and search behavior.
"""

from __future__ import annotations

from hcmai.common.schemas import RetrievalSource
from hcmai.retrieval.embedding.pipeline import TextEmbeddingAdapter
from hcmai.retrieval.retriever.cache import EmbeddingCache
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.dense.retriever import DenseRetriever


class ContextRetriever(DenseRetriever):
    """Retrieve canonical frames through an already-published Context index."""

    def __init__(
        self,
        encoder: TextEmbeddingAdapter,
        index: DenseIndex,
        embedding_cache: EmbeddingCache | None = None,
        prompt_version: str = "query-v1",
    ) -> None:
        """Bind the BGE-compatible query encoder to a loaded Context index."""

        super().__init__(
            encoder,
            index,
            RetrievalSource.CONTEXT,
            embedding_cache,
            prompt_version,
        )
