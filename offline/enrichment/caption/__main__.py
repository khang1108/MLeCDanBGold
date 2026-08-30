"""Command-line entry point for offline caption enrichment."""

from offline.enrichment.pipeline import EnrichmentService

if __name__ == "__main__":
    raise SystemExit(EnrichmentService.run_caption_cli())
