# Kaggle Offline Inference Workers

Each notebook hosts one or more stateless model capabilities behind a named
Cloudflare Tunnel protected by an Access service-token policy.

## Recommended roles

| Role | `MODELS` |
|---|---|
| N1 | `transnet,gebd` |
| N2 | `dino` |
| N3 pool | `caption,ocr` |
| N4 | `visual_emb,text_emb` |
| N5 | `asr,diarization` |

Use two or more N3 replicas when benchmark evidence confirms Florence is the
critical path. All replicas in one endpoint pool must advertise the same exact
checkpoint and revision through `/ready`.

## Start

```bash
MODELS=caption,ocr \
python -m uvicorn kaggle.inference_server:create_kaggle_app \
  --factory --host 127.0.0.1 --port 8100
```

Required configuration paths can be overridden with:

- `HCMAI_PREPARATION_CONFIG`
- `HCMAI_MODEL_CONFIG`
- `HCMAI_ENRICHMENT_CONFIG`

Cloudflare credentials are read by the coordinator from
`HCMAI_CF_ACCESS_CLIENT_ID` and `HCMAI_CF_ACCESS_CLIENT_SECRET`. Do not commit
credentials, tunnel tokens, public URLs, or presigned audio URLs.

## Coordinator

After external raw sync has produced `Lxx/videos` and its inventory:

```bash
python scripts/prepare_group_corpus.py \
  --videos /absolute/path/L21_a/videos \
  --inventory /absolute/path/L21_a/source-manifest.json \
  --cleanup-raw --cleanup-artifacts
```

Run a smoke group without cleanup first. Confirm the S3 `COMMITTED.json` and
rebuild an index from committed vectors before enabling cleanup in production.

