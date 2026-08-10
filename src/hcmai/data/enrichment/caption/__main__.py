"""Run caption enrichment with the configured defaults."""

from hcmai.data.enrichment.pipeline import EnrichmentService

if __name__ == "__main__":
    raise SystemExit(EnrichmentService.run_caption_cli())
