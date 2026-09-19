# Legacy ingestion path

`offline/ingestion/` is retained only for compatibility with the older archive / S3 / BTC workflows.

The V3C local-first preparation path does **not** import its archive, S3, publication, media-info, or yt-dlp orchestration. New VBS work should enter through:

- `offline/corpus/` — local MP4 -> canonical frames
- `offline/clients/` — capability-specific model API roots
- `offline/enrichment/` — local artifact materialization
- `offline/indexes/` — local retrieval bundles
- `offline/pipeline/` — stage orchestration and state
- `scripts/prepare.py` — canonical CLI

Delete or physically move this folder only after the remaining shared artifact models have been detached from it.
