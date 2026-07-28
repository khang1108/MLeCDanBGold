# HCMAI private model service

This folder deploys only GPU inference. It never downloads the keyframe corpus,
FAISS index, mappings, embeddings, or frontend.

## Endpoints

- `GET /health`
- `GET /ready`
- `POST /v1/embeddings/text`
- `POST /v1/rerank` (multipart JPEG candidates)
- `POST /v1/conversation/resolve`

Run locally with one worker so model memory is not duplicated:

```bash
HCMAI_LLM_CONFIG=llm/config.yaml \
PYTHONPATH=src aic/bin/uvicorn hcmai.llm.api:app \
  --host 127.0.0.1 --port 8100 --workers 1
```

## VM deployment

Thunder Compute v2 does not provide systemd or Docker, so this deployment uses
`supervisord`. Connect to the running instance, clone the repository, and
create a root-only secret file:

```bash
git clone https://github.com/khang1108/Multimodal-Question-Answering.git \
  /home/ubuntu/hcmai-bootstrap
sudo install -m 600 /dev/null /root/hcmai-cloudflare.env
sudo nano /root/hcmai-cloudflare.env
```

Populate it from `llm/.env.example` with newly rotated Cloudflare tunnel and API
credentials. Then run:

```bash
sudo bash -c '
  set -a
  source /root/hcmai-cloudflare.env
  set +a
  exec bash "$1"
' _ "$PWD/llm/deploy_cloudflared_private.sh"
```

The first start downloads the configured model checkpoints and may take several
minutes. The script keeps checkpoints under `/opt/hcmai/cache` on the persistent
instance disk.

The script publishes `api.iamphuckhang.dev` to loopback port `8100`. Protect
that hostname with a Cloudflare Access Service Auth policy. The local HCMAI
backend supplies `HCMAI_CF_ACCESS_CLIENT_ID` and
`HCMAI_CF_ACCESS_CLIENT_SECRET`; never put them in the React frontend.

Set `HCMAI_CONVERSATION_MODEL` in `/root/hcmai-cloudflare.env` and rerun the
deployment script to enable the generic Transformers structured conversation
model. Without it, KISC uses its existing deterministic fallback.

Check or restart the processes with:

```bash
sudo supervisorctl status
sudo supervisorctl restart hcmai-llm hcmai-cloudflared
curl -sS http://127.0.0.1:8100/ready | jq
tail -f /opt/hcmai/logs/llm.log /opt/hcmai/logs/cloudflared.log
```

Thunder Compute has no native stopped-instance state. To stop GPU billing while
preserving this setup, create a snapshot, wait until it is ready, and then
delete the instance. Restore a new instance from that snapshot later. If
Supervisor is not running after restoration, start it and restart both
programs:

```bash
sudo supervisord -c /etc/supervisor/supervisord.conf
sudo supervisorctl restart hcmai-llm hcmai-cloudflared
```

## Local search backend

The React UI continues to call `http://127.0.0.1:8000`. On the data machine,
export the values shown in the repository `.env.example` before starting
`hcmai.app`; only that backend calls `https://api.iamphuckhang.dev`. If the
remote encoder is unavailable, configuration permits the existing local encoder
fallback. Reranker failures preserve dense order, and conversation failures use
the existing deterministic KISC fallback.
