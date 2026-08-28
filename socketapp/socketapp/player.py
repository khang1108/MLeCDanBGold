"""Render the standalone HTML player served by SocketApp.

The player owns browser-side playback state only. It seeks by an explicit
non-negative ``timestamp_ms`` value after media metadata is loaded and reports
the media presentation clock in milliseconds while frames are rendered. The
HTTP origin remains responsible for byte-range delivery; it cannot observe a
client's playback position without a separate telemetry request.
"""

from __future__ import annotations

from html import escape


def render_player(
    *,
    video_id: str,
    media_type: str,
    stream_path: str,
    timestamp_ms: int,
    source_fps: float | None,
) -> bytes:
    """Return a self-contained player page for one requested timestamp.

    ``timestamp_ms`` is kept as an integer in the public URL and converted to
    seconds only at the browser media API boundary. The page uses
    ``requestVideoFrameCallback`` when available and falls back to
    ``timeupdate`` so the visible clock follows the actual rendered media time.
    ``source_fps`` comes from the media container when available; a separate
    rendered-FPS counter measures the frames actually presented by the browser.
    """

    safe_video_id = escape(video_id)
    safe_media_type = escape(media_type, quote=True)
    safe_stream_path = escape(stream_path, quote=True)
    source_fps_literal = "null" if source_fps is None else repr(source_fps)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_video_id} · SocketApp</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem auto; max-width: 72rem; padding: 0 1rem; }}
    video {{ background: #000; display: block; max-height: 75vh; width: 100%; }}
    .row {{ align-items: center; display: flex; flex-wrap: wrap; gap: .75rem; margin-top: 1rem; }}
    input {{ font: inherit; max-width: 12rem; padding: .4rem; }}
    output {{ font-variant-numeric: tabular-nums; }}
    .muted {{ color: #aaa; }}
  </style>
</head>
<body>
  <h1>{safe_video_id}</h1>
  <video id="socketapp-video" controls preload="metadata" playsinline>
    <source src="{safe_stream_path}" type="{safe_media_type}">
    Your browser does not support HTML video.
  </video>
  <div class="row">
    <label for="target-ms">Start timestamp (ms)</label>
    <input id="target-ms" type="number" min="0" step="1" value="{timestamp_ms}">
    <button id="seek-button" type="button">Seek and play</button>
  </div>
  <div class="row">
    <span>Current timestamp:</span>
    <output id="current-ms" aria-live="polite">0 ms</output>
    <span class="muted">Duration:</span>
    <output id="duration-ms">loading…</output>
    <span class="muted">Source FPS:</span>
    <output id="source-fps">unknown</output>
    <span class="muted">Rendered FPS:</span>
    <output id="rendered-fps">waiting…</output>
  </div>
  <p id="player-status" class="muted" role="status"></p>
  <script>
    (() => {{
      const video = document.getElementById("socketapp-video");
      const targetInput = document.getElementById("target-ms");
      const seekButton = document.getElementById("seek-button");
      const currentOutput = document.getElementById("current-ms");
      const durationOutput = document.getElementById("duration-ms");
      const sourceFpsOutput = document.getElementById("source-fps");
      const renderedFpsOutput = document.getElementById("rendered-fps");
      const status = document.getElementById("player-status");
      const requestedMs = {timestamp_ms};
      const sourceFps = {source_fps_literal};
      let fpsWindowStart = null;
      let fpsWindowFrames = 0;

      function integerMs(value) {{
        const number = Number(value);
        return Number.isFinite(number) && number >= 0 ? Math.round(number) : 0;
      }}

      function showCurrent(seconds) {{
        currentOutput.value = `${{integerMs(seconds * 1000)}} ms`;
        currentOutput.textContent = currentOutput.value;
      }}

      function showDuration() {{
        if (Number.isFinite(video.duration) && video.duration >= 0) {{
          const durationMs = integerMs(video.duration * 1000);
          durationOutput.value = `${{durationMs}} ms`;
          durationOutput.textContent = durationOutput.value;
        }}
      }}

      function formatFps(value) {{
        return `${{value.toFixed(3).replace(/\\.?0+$/, "")}} fps`;
      }}

      function showSourceFps() {{
        if (Number.isFinite(sourceFps) && sourceFps > 0) {{
          sourceFpsOutput.value = formatFps(sourceFps);
          sourceFpsOutput.textContent = sourceFpsOutput.value;
        }}
      }}

      function countRenderedFrame(now) {{
        if (fpsWindowStart === null) {{
          fpsWindowStart = now;
        }}
        fpsWindowFrames += 1;
        const elapsed = now - fpsWindowStart;
        if (elapsed >= 500) {{
          renderedFpsOutput.value = formatFps(fpsWindowFrames * 1000 / elapsed);
          renderedFpsOutput.textContent = renderedFpsOutput.value;
          fpsWindowStart = now;
          fpsWindowFrames = 0;
        }}
      }}

      function seekToMs(value) {{
        const requested = integerMs(value);
        const durationMs = Number.isFinite(video.duration)
          ? integerMs(video.duration * 1000)
          : null;
        const target = durationMs === null ? requested : Math.min(requested, durationMs);
        video.currentTime = target / 1000;
        targetInput.value = String(target);
        showCurrent(video.currentTime);
      }}

      video.addEventListener("loadedmetadata", () => {{
        showSourceFps();
        showDuration();
        seekToMs(requestedMs);
      }});
      video.addEventListener("durationchange", showDuration);
      video.addEventListener("timeupdate", () => showCurrent(video.currentTime));
      video.addEventListener("error", () => {{
        status.textContent = "The video could not be decoded or loaded.";
      }});
      seekButton.addEventListener("click", () => {{
        seekToMs(targetInput.value);
        video.play().catch(() => {{
          status.textContent = "Timestamp selected; press Play if autoplay is blocked.";
        }});
      }});

      if (typeof video.requestVideoFrameCallback === "function") {{
        const updateRenderedClock = (_now, metadata) => {{
          countRenderedFrame(_now);
          const mediaSeconds = Number.isFinite(metadata.mediaTime)
            ? metadata.mediaTime
            : video.currentTime;
          showCurrent(mediaSeconds);
          video.requestVideoFrameCallback(updateRenderedClock);
        }};
        video.requestVideoFrameCallback(updateRenderedClock);
      }} else {{
        renderedFpsOutput.textContent = "unavailable";
      }}
    }})();
  </script>
</body>
</html>
"""
    return document.encode("utf-8")
