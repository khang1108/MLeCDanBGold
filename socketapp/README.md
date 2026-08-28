# HCMAI SocketApp

SocketApp is a small local video origin for the HCMAI frame inspector. It
serves files from one configured directory over ordinary HTTP, with browser
compatible byte ranges, so seeking does not depend on a YouTube iframe or a
custom client protocol.

The process deliberately does not download videos, run FFmpeg, expose the
filesystem, or read Cloudflare credentials. It loads a validated video catalog
once and uses a bounded number of request threads. Regular-file transfers use
Python's `socket.sendfile` implementation when the platform provides it.

## 1. Put source videos on the local machine

The repository currently contains BTC keyframes, not source videos. Store the
playable source files in a directory such as:

```text
/home/endipi/personal/MLeCDanBGold/data/videos/
├── L21_V001.mp4
├── L21_V002.mp4
└── ...
```

MP4 using H.264/AAC is the safest browser format. The origin also recognizes
WebM, M4V, MOV, MKV, and AVI, although browser codec support for those formats
varies.

If a filename stem is the exact canonical `video_id`, automatic discovery is
enough:

```bash
cd /home/endipi/personal/MLeCDanBGold/socketapp
python3 -m socketapp \
  --video-root /home/endipi/personal/MLeCDanBGold/data/videos
```

For canonical IDs whose local filenames differ, generate and review a
manifest. This is also the recommended production mode:

```bash
python3 scripts/build_manifest.py \
  --video-root /home/endipi/personal/MLeCDanBGold/data/videos \
  --output /home/endipi/personal/MLeCDanBGold/socketapp/videos.json
```

The manifest can be edited into this form when one local file serves a
canonical ID:

```json
{
  "version": 1,
  "videos": [
    {
      "video_id": "L21_V001",
      "path": "L21_V001.mp4",
      "mime_type": "video/mp4"
    }
  ]
}
```

Every manifest path is resolved under `SOCKETAPP_VIDEO_ROOT`; traversal outside
that directory is rejected. IDs are exact and case-sensitive. This prevents a
leaf filename or a request URL from silently replacing a canonical identity.

## 2. Run and verify the origin

Python 3.11 or newer is enough; the service has no third-party runtime
dependency.

```bash
cd /home/endipi/personal/MLeCDanBGold/socketapp
export SOCKETAPP_VIDEO_ROOT=/home/endipi/personal/MLeCDanBGold/data/videos
export SOCKETAPP_MANIFEST=/home/endipi/personal/MLeCDanBGold/socketapp/videos.json
export SOCKETAPP_CORS_ORIGINS=http://127.0.0.1:3000,http://localhost:3000
python3 -m socketapp --host 127.0.0.1 --port 8120
```

Useful checks:

```bash
curl --fail http://127.0.0.1:8120/ready
curl --fail http://127.0.0.1:8120/api/v1/videos/L21_V001
curl --fail -I http://127.0.0.1:8120/api/v1/videos/L21_V001/stream
curl --fail -H 'Range: bytes=0-1023' \
  -o /tmp/socketapp-sample.bin \
  http://127.0.0.1:8120/api/v1/videos/L21_V001/stream
```

The stream endpoint is:

```text
GET /api/v1/videos/{url-encoded-video-id}/stream
HEAD /api/v1/videos/{url-encoded-video-id}/stream
```

It returns `206 Partial Content` for a satisfiable single byte range and
`416 Range Not Satisfiable` with `Content-Range: bytes */{size}` for an invalid
range. `ETag`, `Last-Modified`, and `If-None-Match` are included so browser
revalidation does not resend an unchanged file.

For a standalone browser player with millisecond seeking and a continuously
updated playback clock, use:

```text
GET /api/v1/videos/{url-encoded-video-id}/play?timestamp_ms=5000
HEAD /api/v1/videos/{url-encoded-video-id}/play?timestamp_ms=5000
```

`timestamp_ms` must be one non-negative integer. The page seeks after media
metadata is loaded, displays the actual rendered media timestamp in
milliseconds, and provides a timestamp input for repeated seeks. The catalog
probes the source FPS once at startup with `ffprobe` when it is installed; the
page displays that source FPS separately from the live rendered FPS measured by
the browser. If `ffprobe` is unavailable or the file has no usable rate, the
source FPS is shown as unknown. The raw `/stream` endpoint remains a normal
byte-range media resource; a browser or client owns the playback clock, so the
origin cannot observe a viewer's current timestamp without an explicit
telemetry request.

## 3. Connect Cloudflare Tunnel

The current deployment script uses a remotely managed token-based tunnel. The
new video route must therefore be added to that tunnel's hostname routing in
the Cloudflare dashboard/API:

```text
video.<your-domain> -> http://127.0.0.1:8120
```

Run SocketApp on the same machine as the `cloudflared` connector. A tunnel is
an outbound connector to Cloudflare; it does not make the origin port public.
If the tunnel connector is instead running on another VM, `127.0.0.1` there is
that VM, not this workstation. In that case run the origin on the VM or route
the VM to this machine over a private network.

For a locally managed tunnel, `deploy/cloudflared-config.example.yml` shows
the equivalent ingress rule. Keep the tunnel token/credential in the existing
secret file or supervisor environment; do not copy it into this application.

Put Cloudflare Access or another authentication layer in front of a public
hostname. `SOCKETAPP_CORS_ORIGINS` controls browser read permission only and is
not authentication. If Access is used, its policy must allow the browser's
media `GET`/`HEAD` range requests; a service-token-only policy requiring a
custom `Authorization` header cannot be attached to a native `<video>` URL
without a separate authenticated proxy.

## 4. Use it from another client

Clients that already know the canonical video ID can use the raw stream URL:

```text
GET /api/v1/videos/{url-encoded-video-id}/stream
```

For a browser-native player, use the standalone `/play` URL. For a custom
client, convert the requested integer millisecond timestamp to the client's
media API units only at that API boundary (for example, seconds for HTML
video); keep the application contract in milliseconds.

## 5. Supervisor

`deploy/supervisor.conf.example` is a credential-free process definition. It
keeps the origin on loopback, restarts it if it exits, and makes the video root
explicit. Adjust the user, paths, and log directory before installing it.

## Design boundaries

* The catalog is loaded at startup; restart the process after adding/removing
  videos or regenerate the manifest.
* Only one byte range is served per request. This matches browser video
  seeking and avoids multipart-range buffering.
* Active requests are bounded by `SOCKETAPP_MAX_WORKERS` (32 by default).
* No Cloudflare token, Access secret, AWS key, or YouTube credential is read or
  logged by SocketApp.
