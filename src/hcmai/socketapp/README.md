# Integrated local video origin

`hcmai.socketapp` serves canonical source videos from the main `hcmai.app`
FastAPI process. It validates every file under one configured directory and
supports browser-compatible `GET`/`HEAD` byte ranges for seeking.

It does not download or transcode videos, expose arbitrary filesystem paths,
or manage Cloudflare credentials.

## Configuration

Video serving is disabled unless `SOCKETAPP_VIDEO_ROOT` is set in the
repository-level `.env`. This lets the retrieval backend start normally on
machines that do not store source videos. The backend loads this file before
constructing the catalog, so no shell `export` is required.

```dotenv
SOCKETAPP_VIDEO_ROOT=/absolute/path/to/videos
# Optional when filenames do not equal canonical video_id values:
SOCKETAPP_MANIFEST=/absolute/path/to/videos.json
# Optional; defaults shown:
SOCKETAPP_ALLOW_EMPTY=false
SOCKETAPP_CACHE_CONTROL=public, max-age=3600
```

```bash
PYTHONPATH=.:src aic/bin/python -m uvicorn hcmai.app:app \
  --host 127.0.0.1 --port 8000
```

Without a manifest, supported videos are discovered recursively and each
filename stem becomes its exact, case-sensitive `video_id`. Supported
extensions are MP4, WebM, M4V, MOV, MKV, and AVI; actual browser codec support
still depends on the media encoding.

Generate an explicit manifest when local filenames differ from canonical IDs:

```bash
PYTHONPATH=.:src aic/bin/python -m hcmai.socketapp.build_manifest \
  --video-root /absolute/path/to/videos \
  --output /absolute/path/to/videos.json
```

## Routes

```text
GET  /api/v1/videos/health
GET  /api/v1/videos
GET  /api/v1/videos/{video_id}
GET  /api/v1/videos/{video_id}/stream
HEAD /api/v1/videos/{video_id}/stream
GET  /api/v1/videos/{video_id}/play?timestamp_ms=5000
HEAD /api/v1/videos/{video_id}/play?timestamp_ms=5000
```

The stream endpoint returns `206 Partial Content` for a satisfiable single byte
range and `416 Range Not Satisfiable` otherwise. Metadata responses never
include local paths.

If a separate public hostname is desired, point the existing Cloudflare Tunnel
route at the main backend origin, for example:

```text
stream.example.com -> http://127.0.0.1:8000
```

The tunnel connector and FastAPI process must run on the same machine when the
origin is loopback. Put authentication such as Cloudflare Access in front of a
public hostname; CORS is not authentication.
