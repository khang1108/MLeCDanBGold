# ThunderCompute lifecycle service

This directory owns the laptop-side controller for the disposable GPU
inference VM. It does not contain the Cloudflare token or the private
bootstrap implementation.

```text
Docker Compose profile
  -> tnr create
  -> wait for RUNNING
  -> tnr scp deploy_cloudflared_private.sh 0:/home/ubuntu/
  -> SSH command starts the private bootstrap in tmux
  -> graceful controller stop
  -> tnr delete
```

Thunder's documented CLI exposes `create`, `scp`, `connect`, and `delete`, but
not a non-interactive `tnr exec` command. `launch.sh` therefore uses the SSH
host, port, UUID, and key material made available by `tnr status --json` after
the upload. The remote command is still executed without an interactive
terminal.

## Private bootstrap

Create the ignored runtime file from the safe template and fill in the actual
deployment commands and Cloudflare tunnel secret:

```bash
cp scripts/thundercompute/deploy_cloudflared_private.sh.example \
  scripts/thundercompute/deploy_cloudflared_private.sh
chmod 700 scripts/thundercompute/deploy_cloudflared_private.sh
```

The bootstrap is uploaded to `/home/ubuntu/deploy_cloudflared_private.sh` and
must configure the current `hcmai.thundercompute.server.api` service, start the
Cloudflare tunnel for `api.iamphuckhang.dev`, and verify `/health` and `/ready`.
The launcher never receives the Cloudflare token as a command-line argument.

## Docker lifecycle

The default Compose stack starts backend, LiteLLM, and frontend. The
ThunderCompute controller is an opt-in `thundercompute` profile:

```bash
mkdir -p .secrets .thundercompute
printf '%s\n' 'your-thunder-api-token' > .secrets/tnr_api_token

# Set these paths in .env, or export them for the command.
export TNR_API_TOKEN_FILE=.secrets/tnr_api_token
export HCMAI_THUNDER_DEPLOY_SCRIPT=./scripts/thundercompute/deploy_cloudflared_private.sh

docker compose --profile thundercompute build thundercompute
docker compose --profile thundercompute up thundercompute
```

Run the normal application stack in another terminal when needed:

```bash
docker compose up -d
```

Use `docker compose stop thundercompute` or `docker compose down` to send a
graceful signal. The controller trap then calls `tnr delete` and clears
`.thundercompute/instance-id`. `docker kill` defaults to `SIGKILL`, which no
process can catch; after a forced kill run:

```bash
TNR_API_TOKEN_FILE=.secrets/tnr_api_token \
  bash scripts/thundercompute/delete.sh
```

Do not set `restart: always` for this controller: restarting it after a clean
exit would create another billable VM.

## Host-side operation

The same controller can run without Docker when `tnr`, Python 3, and OpenSSH
are installed on the laptop:

```bash
TNR_API_TOKEN_FILE=.secrets/tnr_api_token \
  HCMAI_THUNDER_DEPLOY_SCRIPT=./scripts/thundercompute/deploy_cloudflared_private.sh \
  bash scripts/thundercompute/launch.sh --gpu l40 --
```

Press Ctrl-C for cleanup. Use `--keep` only when the instance is intentionally
managed separately, and use `--instance ID --delete-reused` only when the
controller owns that existing instance.
