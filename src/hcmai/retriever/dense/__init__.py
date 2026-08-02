"""Dense encoding, exact FAISS indexing, and online retrieval."""

from hcmai.retriever.dense.bge import BGETextEncoder
from hcmai.retriever.dense.encoder import (
    DenseEncoder,
    TextEncoder,
    create_text_encoder,
)
from hcmai.retriever.dense.index import (
    INDEX_FILENAME,
    MAPPING_FILENAME,
    METADATA_FILENAME,
    DenseIndex,
)
from hcmai.retriever.dense.models import EncodingStats, IndexMetadata
from hcmai.retriever.dense.retriever import DenseRetriever

__all__ = [
    "DenseEncoder",
    "BGETextEncoder",
    "create_text_encoder",
    "DenseIndex",
    "DenseRetriever",
    "EncodingStats",
    "INDEX_FILENAME",
    "IndexMetadata",
    "MAPPING_FILENAME",
    "METADATA_FILENAME",
    "TextEncoder",
]
