"""Small Streamlit dashboard for measuring S3 video streaming latency.

Run from the repository root with::

    aic/bin/python -m streamlit run scripts/s3_streaming_latency_app.py

The server-side probe measures a fixed byte range. The embedded browser player
reports the latency users actually experience while the browser chooses its own
range requests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import os
from time import perf_counter
from typing import Any


DEFAULT_BUCKET = "s3-hcmai-mlecdanbgold-759712674669-ap-southeast-2-an"
DEFAULT_KEY = "data/Videos_L21_a/videos/L21_V001.mp4"
DEFAULT_REGION = "ap-southeast-2"


@dataclass(frozen=True, slots=True)
class RangeProbeResult:
    """Timing and response metadata for one S3 byte-range request."""

    head_ms: float
    response_headers_ms: float
    time_to_first_byte_ms: float
    total_ms: float
    bytes_read: int
    throughput_mbps: float
    object_size_bytes: int
    content_range: str | None
    content_type: str | None

    def as_row(self, attempt: int) -> dict[str, Any]:
        return {"attempt": attempt, **asdict(self)}


def make_s3_client(region: str):
    """Create an S3 client using the standard AWS credential chain."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as error:
        raise RuntimeError(
            'AWS dependencies are missing; install with: pip install -e ".[s3-streaming-ui]"'
        ) from error

    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=os.getenv("HCMAI_S3_ENDPOINT_URL"),
        config=Config(
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 1, "mode": "standard"},
        ),
    )


def probe_range(
    client: Any,
    *,
    bucket: str,
    key: str,
    range_bytes: int,
    chunk_bytes: int = 64 * 1024,
) -> RangeProbeResult:
    """Measure HEAD, headers, first byte, and full download for one range."""
    if range_bytes <= 0:
        raise ValueError("range_bytes must be positive")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    head_started = perf_counter()
    metadata = client.head_object(Bucket=bucket, Key=key)
    head_ms = (perf_counter() - head_started) * 1_000

    request_started = perf_counter()
    response = client.get_object(
        Bucket=bucket,
        Key=key,
        Range=f"bytes=0-{range_bytes - 1}",
    )
    headers_at = perf_counter()
    body = response["Body"]
    first_byte_at: float | None = None
    bytes_read = 0
    try:
        while True:
            chunk = body.read(chunk_bytes)
            if not chunk:
                break
            if first_byte_at is None:
                first_byte_at = perf_counter()
            bytes_read += len(chunk)
    finally:
        body.close()
    completed_at = perf_counter()

    if first_byte_at is None:
        first_byte_at = completed_at
    total_seconds = max(completed_at - request_started, 1e-9)
    return RangeProbeResult(
        head_ms=head_ms,
        response_headers_ms=(headers_at - request_started) * 1_000,
        time_to_first_byte_ms=(first_byte_at - request_started) * 1_000,
        total_ms=total_seconds * 1_000,
        bytes_read=bytes_read,
        throughput_mbps=(bytes_read * 8) / total_seconds / 1_000_000,
        object_size_bytes=int(metadata.get("ContentLength", 0)),
        content_range=response.get("ContentRange"),
        content_type=response.get("ContentType"),
    )


def build_presigned_video_url(
    client: Any,
    *,
    bucket: str,
    key: str,
    expires_seconds: int,
) -> str:
    """Sign a browser-playable GET without changing the object's metadata."""
    if not 60 <= expires_seconds <= 3_600:
        raise ValueError("expires_seconds must be between 60 and 3600")
    return client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ResponseContentType": "video/mp4",
        },
        ExpiresIn=expires_seconds,
    )


def browser_player_html(url: str) -> str:
    """Build a click-to-load player with startup and rebuffer timings."""
    safe_url = html.escape(url, quote=True)
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ margin: 0; color: #e6edf3; background: #0e1117; font: 14px sans-serif; }}
    #player {{ position: relative; min-height: 300px; background: black; border-radius: 8px; overflow: hidden; }}
    video {{ display: block; width: 100%; min-height: 300px; max-height: 430px; background: black; }}
    #start {{
      position: absolute; inset: 0; width: 100%; border: 0; cursor: pointer;
      color: white; background: #000; font-size: 22px; font-weight: 700;
    }}
    #start:hover {{ background: #090d13; }}
    #status {{ margin: 10px 0; font-weight: 600; }}
    #live {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 10px 0; }}
    .metric {{ padding: 8px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; }}
    .metric b {{ display: block; margin-top: 3px; font-size: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ padding: 5px 8px; border-bottom: 1px solid #30363d; text-align: left; }}
    .hint {{ color: #9da7b3; font-size: 12px; }}
  </style>
</head>
<body>
  <div id="player">
    <video id="video" controls muted playsinline preload="none"></video>
    <button id="start" type="button" data-url="{safe_url}">▶ Click to load and play video</button>
  </div>
  <div id="status">No video request has been sent.</div>
  <div id="live">
    <div class="metric">Startup latency<b id="startup">—</b></div>
    <div class="metric">Buffered ahead<b id="buffer">0.00 s</b></div>
    <div class="metric">Playback time<b id="playback">0.00 s</b></div>
    <div class="metric">Rebuffer count<b id="rebuffer-count">0</b></div>
    <div class="metric">Current rebuffer<b id="current-rebuffer">0 ms</b></div>
    <div class="metric">Total rebuffer<b id="total-rebuffer">0 ms</b></div>
  </div>
  <table><thead><tr><th>Browser event</th><th>Elapsed from click</th></tr></thead><tbody id="events"></tbody></table>
  <p class="hint">Nothing is downloaded before the click. During playback, buffer and rebuffer metrics update live.</p>
  <script>
    const video = document.getElementById("video");
    const startButton = document.getElementById("start");
    const rows = document.getElementById("events");
    const status = document.getElementById("status");
    const startup = document.getElementById("startup");
    const buffer = document.getElementById("buffer");
    const playback = document.getElementById("playback");
    const rebufferCountElement = document.getElementById("rebuffer-count");
    const currentRebufferElement = document.getElementById("current-rebuffer");
    const totalRebufferElement = document.getElementById("total-rebuffer");
    let started = null;
    let firstPlayingAt = null;
    let rebufferStartedAt = null;
    let totalRebufferMs = 0;
    let rebufferCount = 0;
    const seen = new Set();
    function record(name, detail = "") {{
      if (started === null) return;
      const elapsed = performance.now() - started;
      if (!["waiting", "stalled", "error", "rebuffer-ended"].includes(name) && seen.has(name)) return;
      seen.add(name);
      const row = document.createElement("tr");
      row.innerHTML = `<td>${{name}}${{detail ? " — " + detail : ""}}</td><td>${{elapsed.toFixed(1)}} ms</td>`;
      rows.appendChild(row);
      status.textContent = `${{name}} at ${{elapsed.toFixed(1)}} ms`;
    }}
    ["loadstart", "durationchange", "loadedmetadata", "loadeddata", "canplay", "progress", "stalled"]
      .forEach(name => video.addEventListener(name, () => record(name)));
    video.addEventListener("error", () => record("error", video.error ? video.error.message : "unknown media error"));
    video.addEventListener("waiting", () => {{
      record("waiting");
      if (firstPlayingAt !== null && !video.paused && rebufferStartedAt === null) {{
        rebufferStartedAt = performance.now();
        rebufferCount += 1;
        rebufferCountElement.textContent = String(rebufferCount);
      }}
    }});
    video.addEventListener("playing", () => {{
      record("playing");
      const now = performance.now();
      if (firstPlayingAt === null) {{
        firstPlayingAt = now;
        startup.textContent = `${{(now - started).toFixed(1)}} ms`;
      }}
      if (rebufferStartedAt !== null) {{
        const duration = now - rebufferStartedAt;
        totalRebufferMs += duration;
        record("rebuffer-ended", `${{duration.toFixed(1)}} ms`);
        rebufferStartedAt = null;
      }}
    }});
    startButton.addEventListener("click", async () => {{
      startButton.hidden = true;
      rows.innerHTML = "";
      seen.clear();
      started = performance.now();
      status.textContent = "Clicked: attaching URL and starting request…";
      record("click");
      video.src = startButton.dataset.url;
      video.load();
      if ("requestVideoFrameCallback" in video) {{
        video.requestVideoFrameCallback(() => record("first-video-frame"));
      }}
      try {{
        await video.play();
      }} catch (error) {{
        record("play-rejected", error.message);
      }}
    }});
    setInterval(() => {{
      playback.textContent = `${{video.currentTime.toFixed(2)}} s`;
      let bufferedAhead = 0;
      for (let index = 0; index < video.buffered.length; index += 1) {{
        if (video.buffered.start(index) <= video.currentTime && video.currentTime <= video.buffered.end(index)) {{
          bufferedAhead = video.buffered.end(index) - video.currentTime;
          break;
        }}
      }}
      buffer.textContent = `${{Math.max(0, bufferedAhead).toFixed(2)}} s`;
      const currentRebufferMs = rebufferStartedAt === null ? 0 : performance.now() - rebufferStartedAt;
      currentRebufferElement.textContent = `${{currentRebufferMs.toFixed(0)}} ms`;
      totalRebufferElement.textContent = `${{(totalRebufferMs + currentRebufferMs).toFixed(0)}} ms`;
    }}, 100);
  </script>
</body>
</html>
"""


def main() -> None:
    try:
        import pandas as pd
        import streamlit as st
        import streamlit.components.v1 as components
    except ImportError as error:
        raise RuntimeError(
            'UI dependencies are missing; install with: pip install -e ".[s3-streaming-ui]"'
        ) from error

    st.set_page_config(page_title="S3 video latency", page_icon="🎬", layout="wide")
    st.title("S3 video streaming latency")
    st.caption(
        "Server probe measures a fixed byte range; browser playback measures actual startup and buffering. "
        "AWS credentials stay on the Streamlit server."
    )

    with st.sidebar:
        st.header("S3 object")
        bucket = st.text_input("Bucket", DEFAULT_BUCKET)
        key = st.text_input("Key", DEFAULT_KEY)
        region = st.text_input("Region", DEFAULT_REGION)
        range_mib = st.number_input("Probe range (MiB)", min_value=0.001, max_value=64.0, value=1.0)
        attempts = st.number_input("Probe attempts", min_value=1, max_value=20, value=3, step=1)
        expires_seconds = st.number_input(
            "Player URL lifetime (seconds)", min_value=60, max_value=3_600, value=900, step=60
        )

    if not bucket.strip() or not key.strip() or not region.strip():
        st.error("Bucket, key, and region are required.")
        return

    try:
        client = make_s3_client(region.strip())
    except Exception as error:
        st.error(str(error))
        return

    probe_column, player_column = st.columns(2)
    with probe_column:
        st.subheader("1. Server-side range probe")
        if st.button("Run latency probe", type="primary", use_container_width=True):
            rows: list[dict[str, Any]] = []
            try:
                with st.spinner("Reading S3 byte ranges…"):
                    for attempt in range(1, int(attempts) + 1):
                        result = probe_range(
                            client,
                            bucket=bucket.strip(),
                            key=key.strip(),
                            range_bytes=max(1, int(float(range_mib) * 1024 * 1024)),
                        )
                        rows.append(result.as_row(attempt))
            except Exception as error:
                st.exception(error)
            else:
                frame = pd.DataFrame(rows)
                metric_columns = st.columns(3)
                metric_columns[0].metric("Median TTFB", f"{frame['time_to_first_byte_ms'].median():.1f} ms")
                metric_columns[1].metric("Median total", f"{frame['total_ms'].median():.1f} ms")
                metric_columns[2].metric("Median throughput", f"{frame['throughput_mbps'].median():.1f} Mbps")
                st.dataframe(frame, hide_index=True, use_container_width=True)
                st.caption(
                    f"Object size: {rows[0]['object_size_bytes'] / 1_000_000:.1f} MB · "
                    f"S3 response type: {rows[0]['content_type']} · {rows[0]['content_range']}"
                )

    with player_column:
        st.subheader("2. Browser playback probe")
        st.write("No video bytes are requested until you click the black player below.")
        try:
            signed_url = build_presigned_video_url(
                client,
                bucket=bucket.strip(),
                key=key.strip(),
                expires_seconds=int(expires_seconds),
            )
        except Exception as error:
            st.exception(error)
        else:
            st.warning("The embedded presigned URL temporarily grants read access. Do not share the page source.")
            components.html(browser_player_html(signed_url), height=760)


if __name__ == "__main__":
    main()
