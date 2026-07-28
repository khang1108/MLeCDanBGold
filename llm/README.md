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

## One-time private bootstrap setup

Thunder Compute v2 does not provide systemd or Docker, so deployment uses
`supervisord`. `deploy_cloudflared_private.sh` is intentionally ignored by Git.
Paste the remotely-managed `eyJ...` token once into its
`CLOUDFLARE_TUNNEL_TOKEN` variable and keep this private copy on the local
machine.

Configure the tunnel's Published application route once in Cloudflare:
`api.iamphuckhang.dev` to `http://localhost:8100`. The tunnel, DNS route, and
Access policy remain in Cloudflare after a Thunder instance is deleted.

## Every throwaway VM

From the local machine, copy the private bootstrap to the new instance and
connect:

```bash
tnr scp llm/deploy_cloudflared_private.sh 0:/home/ubuntu/
tnr connect 0
```

On the VM, run exactly one command:

```bash
sudo bash /home/ubuntu/deploy_cloudflared_private.sh
```

The bootstrap clones the configured repository and branch into
`/opt/hcmai/repo`, installs all required packages, downloads model checkpoints,
starts the model API and tunnel, and verifies both processes. No manual clone,
environment file, Tunnel ID, Cloudflare API token, or DNS API setup is needed
on the VM. All repository commands run as the `hcmai` service user, so no Git
login, global `safe.directory`, `user.name`, or `user.email` configuration is
required for a public repository.

On every bootstrap run, the deployment checkout is fetched and reset to the
latest commit of the configured `origin` branch. Tracked edits made directly
inside `/opt/hcmai/repo` are intentionally discarded; make changes locally,
push them to `main`, and rerun the bootstrap to update and restart the service.
The application virtual environment and downloaded checkpoint cache are reused.

The first start downloads the configured model checkpoints and may take several
minutes. The script keeps checkpoints under `/opt/hcmai/cache` on the persistent
instance disk.

In the tunnel's Cloudflare dashboard, add a Published application route from
`api.iamphuckhang.dev` to `http://localhost:8100`. Protect that hostname with a
Cloudflare Access Service Auth policy. The local HCMAI backend supplies
`HCMAI_CF_ACCESS_CLIENT_ID` and
`HCMAI_CF_ACCESS_CLIENT_SECRET`; never put them in the React frontend.

The pinned conversation checkpoint in `llm/config.yaml` is used when
`HCMAI_CONVERSATION_MODEL` is empty. Set that variable in the private bootstrap
only to override the checkpoint. To disable hosted conversation inference, set
`conversation.checkpoint` to `null`; KISC will then use its deterministic
fallback.

GLM-4.1V-9B-Thinking is loaded through its official multimodal processor and
conditional-generation class. BF16 weights require substantial VRAM alongside
the embedding and reranking models; use a sufficiently large GPU or add an
explicit quantized profile before deploying on a 24 GB device.

Check or restart the processes with:

```bash
sudo supervisorctl status
sudo supervisorctl restart hcmai-llm hcmai-cloudflared
curl -sS http://127.0.0.1:8100/ready | jq
tail -f /opt/hcmai/logs/llm.log /opt/hcmai/logs/cloudflared.log
```

The bootstrap installs Supervisor 4.3.0 in `/opt/hcmai/supervisor` and exposes
its commands through `/usr/local/bin`. This avoids the Ubuntu Supervisor 4.2.1
package importing the removed Python 3.12 `asynchat` module. If an older VM
shows `ModuleNotFoundError: No module named 'asynchat'`, upload the latest local
private bootstrap again and rerun it; the operation reuses the existing
repository and application virtual environment.

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
