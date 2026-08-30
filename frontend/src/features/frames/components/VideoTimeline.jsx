import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { keyframeUrl } from "../../../api/keyframes";
import { getRaw1FpsFrameId } from "../videoSource";

const clamp = (value, minimum, maximum) => Math.min(Math.max(value, minimum), maximum);

export const formatVideoTime = (seconds) => {
  const safeSeconds = Number.isFinite(seconds) && seconds >= 0 ? Math.floor(seconds) : 0;
  const minutes = Math.floor(safeSeconds / 60);
  const remainingSeconds = safeSeconds % 60;
  if (minutes < 60) {
    return `${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
  }

  const hours = Math.floor(minutes / 60);
  return `${String(hours).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
};

const VideoTimeline = ({
  videoId,
  videoRef,
  currentTime,
  duration,
  onSeek,
  onTogglePlayback,
}) => {
  const timelineRef = useRef(null);
  const previewUrlCache = useRef(new Map());
  const [hoverPreview, setHoverPreview] = useState(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [volume, setVolume] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : 0;
  const safeCurrentTime = clamp(
    Number.isFinite(currentTime) && currentTime >= 0 ? currentTime : 0,
    0,
    safeDuration || 1,
  );
  const progressPercent = safeDuration > 0
    ? (safeCurrentTime / safeDuration) * 100
    : 0;

  useEffect(() => {
    const video = videoRef?.current;
    if (!video) return undefined;

    const syncPlaybackState = () => {
      setIsPlaying(!video.paused && !video.ended);
    };

    syncPlaybackState();
    video.addEventListener?.("play", syncPlaybackState);
    video.addEventListener?.("pause", syncPlaybackState);
    video.addEventListener?.("ended", syncPlaybackState);
    return () => {
      video.removeEventListener?.("play", syncPlaybackState);
      video.removeEventListener?.("pause", syncPlaybackState);
      video.removeEventListener?.("ended", syncPlaybackState);
    };
  }, [videoRef]);

  const previewForEvent = useCallback((event) => {
    if (!safeDuration || !timelineRef.current) return null;

    const bounds = timelineRef.current.getBoundingClientRect();
    if (!bounds.width) return null;

    const percent = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
    const time = percent * safeDuration;
    const lastFrameSecond = Math.max(Math.ceil(safeDuration) - 1, 0);
    const second = clamp(Math.floor(time), 0, lastFrameSecond);
    const frameId = getRaw1FpsFrameId(videoId, second * 1000);
    if (!frameId) return null;

    let url = previewUrlCache.current.get(second);
    if (!url) {
      url = keyframeUrl(frameId);
      previewUrlCache.current.set(second, url);
    }

    return { percent: percent * 100, second, time, url };
  }, [safeDuration, videoId]);

  const handleTimelineHover = useCallback((event) => {
    const nextPreview = previewForEvent(event);
    if (!nextPreview) return;
    setPreviewFailed(false);
    setHoverPreview(nextPreview);
  }, [previewForEvent]);

  const handleTimelineLeave = useCallback(() => {
    setHoverPreview(null);
    setPreviewFailed(false);
  }, []);

  const handleSeek = useCallback((event) => {
    const nextTime = Number(event.target.value);
    if (Number.isFinite(nextTime)) onSeek?.(nextTime);
  }, [onSeek]);

  const togglePlayback = useCallback(() => {
    if (onTogglePlayback) {
      onTogglePlayback();
      return;
    }

    const video = videoRef?.current;
    if (!video) return;

    if (video.paused || video.ended) {
      const playResult = video.play?.();
      playResult?.catch?.(() => setIsPlaying(false));
    } else {
      video.pause?.();
    }
  }, [onTogglePlayback, videoRef]);

  const toggleMute = useCallback(() => {
    const video = videoRef?.current;
    if (!video) return;
    const nextMuted = !video.muted;
    video.muted = nextMuted;
    setIsMuted(nextMuted);
  }, [videoRef]);

  const handleVolumeChange = useCallback((event) => {
    const nextVolume = clamp(Number(event.target.value), 0, 1);
    const video = videoRef?.current;
    if (!video || !Number.isFinite(nextVolume)) return;
    video.volume = nextVolume;
    video.muted = nextVolume === 0;
    setVolume(nextVolume);
    setIsMuted(nextVolume === 0);
  }, [videoRef]);

  const toggleFullscreen = useCallback(() => {
    const shell = videoRef?.current?.closest(".modal-video-shell");
    if (!shell) return;

    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      shell.requestFullscreen?.();
    }
  }, [videoRef]);

  useEffect(() => {
    const syncFullscreenState = () => {
      setIsFullscreen(document.fullscreenElement === videoRef?.current?.closest(".modal-video-shell"));
    };
    document.addEventListener("fullscreenchange", syncFullscreenState);
    return () => document.removeEventListener("fullscreenchange", syncFullscreenState);
  }, [videoRef]);

  const previewLabel = useMemo(
    () => (hoverPreview ? `Preview at ${formatVideoTime(hoverPreview.time)}` : ""),
    [hoverPreview],
  );

  return (
    <div className="modal-video-controls" aria-label="Video controls">
      <div
        ref={timelineRef}
        className="modal-timeline-track"
        data-testid="video-timeline-track"
        onMouseMove={handleTimelineHover}
        onPointerMove={handleTimelineHover}
        onMouseLeave={handleTimelineLeave}
        onPointerLeave={handleTimelineLeave}
      >
        {hoverPreview && (
          <div
            className="modal-timeline-preview"
            style={{ left: `${hoverPreview.percent}%` }}
            role="status"
            aria-label={previewLabel}
          >
            {!previewFailed ? (
              <img
                src={hoverPreview.url}
                alt={previewLabel}
                onError={() => setPreviewFailed(true)}
              />
            ) : (
              <div className="modal-timeline-preview-fallback">Preview unavailable</div>
            )}
            <span>{formatVideoTime(hoverPreview.time)}</span>
          </div>
        )}
        <input
          className="modal-timeline-input"
          type="range"
          min="0"
          max={safeDuration || 1}
          step="0.01"
          value={safeCurrentTime}
          disabled={!safeDuration}
          aria-label="Video timeline"
          onChange={handleSeek}
          style={{ "--timeline-progress": `${progressPercent}%` }}
        />
      </div>
      <div className="modal-video-control-row">
        <button
          type="button"
          className="modal-video-play-button"
          onClick={togglePlayback}
          aria-label={isPlaying ? "Pause video" : "Play video"}
        >
          {isPlaying ? "Ⅱ" : "▶"}
        </button>
        <span className="modal-video-time-readout">
          {formatVideoTime(safeCurrentTime)} / {formatVideoTime(safeDuration)}
        </span>
        <div className="modal-video-volume-group">
          <button
            type="button"
            className="modal-video-icon-button"
            onClick={toggleMute}
            aria-label={isMuted ? "Unmute video" : "Mute video"}
          >
            {isMuted || volume === 0 ? "🔇" : "🔊"}
          </button>
          <input
            className="modal-video-volume-input"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={isMuted ? 0 : volume}
            aria-label="Video volume"
            onChange={handleVolumeChange}
            style={{ "--volume-progress": `${(isMuted ? 0 : volume) * 100}%` }}
          />
        </div>
        <button
          type="button"
          className="modal-video-icon-button modal-video-fullscreen-button"
          onClick={toggleFullscreen}
          aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
        >
          ⛶
        </button>
      </div>
    </div>
  );
};

export default VideoTimeline;
