# HCMAI Thunder GPU inference

This directory deploys the batch-oriented caption, OCR, and ASR inference
services on ThunderCompute. Corpus artifacts, indexes, mappings, and the web
frontend are not hosted on the GPU VM.

## Public services

One remotely managed Cloudflare tunnel exposes two local Uvicorn processes:

| Public hostname | Local origin | Capabilities |
| --- | --- | --- |
| `api.iamphuckhang.dev` | `http://127.0.0.1:8100` | Caption and OCR |
| `asr.iamphuckhang.dev` | `http://127.0.0.1:8101` | ASR/transcripts |

Configure both Published application routes on the Cloudflare tunnel before
deploying. Protect both hostnames with the same Cloudflare Access Service Auth
policy when private access is required.

The main inference endpoints are:

- `POST https://api.iamphuckhang.dev/v1/captions`
- `POST https://api.iamphuckhang.dev/v1/enrichment/ocr`
- `POST https://asr.iamphuckhang.dev/v1/transcripts/asr`
- `GET /health`, `GET /ready`, and `GET /docs` on both hostnames

Caption and OCR are batched and reuse the same identically pinned Florence
model in one process. ASR runs in its own process so its model lifecycle and
readiness are isolated.

## Launch a Thunder instance

All three capabilities default to enabled:

```bash
llm/launch_thunder_instance.sh --gpu l40 --token "$TNR_TOKEN" --
```

Arguments after `--` are forwarded to the private bootstrap. Individual
capabilities can be disabled when a VM should host only one service group:

```bash
# Caption + OCR only
llm/launch_thunder_instance.sh --gpu l40 --token "$TNR_TOKEN" -- \
  --asr false

# ASR only
llm/launch_thunder_instance.sh --gpu l40 --token "$TNR_TOKEN" -- \
  --caption false --ocr false
```

The launcher creates or reuses a Thunder instance, uploads
`deploy_cloudflared_private.sh`, and starts it in a detached tmux session. The
private bootstrap clones the configured repository into `/opt/hcmai/repo`,
installs the inference dependencies, starts Supervisor, starts the enabled
Uvicorn processes, starts `cloudflared`, and checks local and public readiness.

## Manual deployment

Copy the ignored private bootstrap to the VM and run it:

```bash
tnr scp llm/deploy_cloudflared_private.sh 0:/home/ubuntu/
tnr connect 0
sudo bash /home/ubuntu/deploy_cloudflared_private.sh
```

The bootstrap is intentionally ignored because it contains the remotely
managed tunnel token and may contain Cloudflare Access service credentials.
Never commit that private copy.

## Operations

```bash
sudo supervisorctl status
sudo supervisorctl restart hcmai-caption-ocr hcmai-asr hcmai-cloudflared

curl -sS http://127.0.0.1:8100/ready | jq
curl -sS http://127.0.0.1:8101/ready | jq

tail -f \
  /opt/hcmai/logs/caption-ocr.log \
  /opt/hcmai/logs/asr.log \
  /opt/hcmai/logs/cloudflared.log
```

The first start downloads pinned model checkpoints into `/opt/hcmai/cache`.
Supervisor runs one worker per service to prevent duplicate GPU model copies.
Re-running the bootstrap updates the checkout to the current configured branch
and restarts the services.
