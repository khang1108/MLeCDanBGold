"""Offline ingestion, enrichment, and artifact-building entry points."""

# Keep this package initializer deliberately import-free: offline builders
# must never pull serving orchestration or HTTP contracts into runtime imports.
__all__: list[str] = []
