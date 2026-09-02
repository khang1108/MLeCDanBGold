"""Load the optional Filter catalog independently from Search/FAISS startup.

This module resolves configuration and environment paths once. It never builds
or repairs the offline catalog and never changes Search readiness.
"""

from __future__ import annotations

import os

from pathlib import Path

from dotenv import load_dotenv

from hcmai.common.config import AppConfig, REPOSITORY_ROOT, resolve_repository_path

from .catalog import FilterCatalog, FilterCatalogUnavailableError
from .service import FilterService


def load_filter_service(messages: list[str]) -> FilterService | None:
    """Open configured Filter SQLite or return a logged optional degradation."""

    load_dotenv(REPOSITORY_ROOT / ".env", override=False)
    config_path = resolve_repository_path(
        os.getenv("HCMAI_CONFIG_PATH", "configs/baseline.yaml")
    )
    if not config_path.is_file():
        messages.append(f"Filter config is not available at {config_path}")
        return None
    settings = AppConfig.from_yaml(config_path)
    catalog_path = resolve_repository_path(
        os.getenv("HCMAI_FILTER_CATALOG_PATH", str(settings.filter.catalog_path))
    )
    try:
        catalog = FilterCatalog.open(
            catalog_path,
            pool_size=settings.filter.connection_pool_size,
            cache_kib=settings.filter.sqlite_cache_kib,
        )
    except FilterCatalogUnavailableError as error:
        messages.append(str(error))
        return None
    return FilterService(catalog)

