"""Exact disk-backed metadata filtering for canonical HCMAI frames.

The package owns Filter-specific normalization and query services. It does not
perform semantic retrieval or mutate canonical frame identity.
"""

from .catalog import (
    CatalogAvailability,
    FilterCatalog,
    FilterCatalogCorruptError,
    FilterCatalogInfo,
    FilterCatalogUnavailableError,
)
from .normalization import normalize_filter_text

__all__ = [
    "CatalogAvailability",
    "FilterCatalog",
    "FilterCatalogCorruptError",
    "FilterCatalogInfo",
    "FilterCatalogUnavailableError",
    "normalize_filter_text",
]
