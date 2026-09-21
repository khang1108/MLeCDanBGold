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
  onSubmit,
  isSubmitting = false,
  onSelectCandidate,
  isCandidateSelected = false,
  onToggleTheater,
  isTheaterMode = false,
  onToggleFitMode,
  fitMode = 'contain',
}) => {
  const timelineRef = useRef(null);
  const previewUrlCache = useRef(new Map());
  const [hoverPreview, setHoverPreview] = useState(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [volume, setVolume] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);
  const [showRemainingTime, setShowRemainingTime] = useState(false);
  const speedMenuRef = useRef(null);

  const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 2];

  const handleSelectSpeed = useCallback((rate) => {
    const video = videoRef?.current;
    if (video) {
      video.playbackRate = rate;
    }
    setPlaybackRate(rate);
    setIsSpeedMenuOpen(false);
  }, [videoRef]);

  useEffect(() => {
    if (!isSpeedMenuOpen) return;
    const handleClickOutside = (e) => {
      if (speedMenuRef.current && !speedMenuRef.current.contains(e.target)) {
        setIsSpeedMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isSpeedMenuOpen]);

  const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : 0;
  const safeCurrentTime = clamp(
    Number.isFinite(currentTime) && currentTime >= 0 ? currentTime : 0,
    0,
    safeDuration || 1,
  );
  const progressPercent = safeDuration > 0
    ? (safeCurrentTime / safeDuration) * 100
    : 0;

  const handleStepFrame = useCallback((direction) => {
    const video = videoRef?.current;
    if (!video) return;
    video.pause?.();
    const step = 0.04 * direction; // ~1 frame at 25fps
    const currentTimeVal = Number.isFinite(video.currentTime) ? video.currentTime : safeCurrentTime;
    const nextTime = Math.round(clamp(currentTimeVal + step, 0, safeDuration) * 1000) / 1000;
    video.currentTime = nextTime;
    onSeek?.(nextTime);
  }, [videoRef, safeCurrentTime, safeDuration, onSeek]);

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
        <div className="modal-video-left-controls">
          <button
            type="button"
            className="modal-video-play-button"
            onClick={togglePlayback}
            aria-label={isPlaying ? "Pause video" : "Play video"}
            title={isPlaying ? "Pause (k / Space)" : "Play (k / Space)"}
          >
            {isPlaying ? "Ⅱ" : "▶"}
          </button>
          <button
            type="button"
            className="modal-video-icon-button modal-video-step-button"
            onClick={() => handleStepFrame(-1)}
            title="Previous frame ( , )"
            aria-label="Previous frame"
          >
            ⏮
          </button>
          <button
            type="button"
            className="modal-video-icon-button modal-video-step-button"
            onClick={() => handleStepFrame(1)}
            title="Next frame ( . )"
            aria-label="Next frame"
          >
            ⏭
          </button>
          <div className="modal-video-volume-group">
            <button
              type="button"
              className="modal-video-icon-button"
              onClick={toggleMute}
              aria-label={isMuted ? "Unmute video" : "Mute video"}
              title={isMuted ? "Unmute (m)" : "Mute (m)"}
            >
              {isMuted || volume === 0
                ? "🔇"
                : volume < 0.35
                  ? "🔈"
                  : volume < 0.75
                    ? "🔉"
                    : "🔊"}
            </button>
            <div className="modal-video-volume-slider-wrap">
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
          </div>
          <span
            className="modal-video-time-readout"
            onClick={() => setShowRemainingTime((prev) => !prev)}
            title="Click to toggle remaining time"
            style={{ cursor: "pointer", userSelect: "none" }}
          >
            {showRemainingTime
              ? `-${formatVideoTime(Math.max(0, safeDuration - safeCurrentTime))} / ${formatVideoTime(safeDuration)}`
              : `${formatVideoTime(safeCurrentTime)} / ${formatVideoTime(safeDuration)}`}
          </span>
        </div>

        <div className="modal-video-right-controls">
          {onSelectCandidate && (
            <button
              type="button"
              className={`modal-video-select-button ${isCandidateSelected ? 'is-selected' : ''}`}
              onClick={onSelectCandidate}
              title={isCandidateSelected ? "Deselect candidate" : "Select candidate"}
              aria-label={isCandidateSelected ? "Deselect candidate" : "Select candidate"}
            >
              {isCandidateSelected ? "✓ Selected" : "+ Select"}
            </button>
          )}
          {onSubmit && (
            <button
              type="button"
              className="modal-video-submit-button"
              onClick={onSubmit}
              disabled={isSubmitting}
              title={`Submit at ${Math.round(safeCurrentTime * 1000)} ms`}
              aria-label="Submit current playback time to DRES"
            >
              ↗ Submit
            </button>
          )}
          <div className="modal-video-speed-group" ref={speedMenuRef}>
            <button
              type="button"
              className={`modal-video-icon-button modal-video-speed-button ${playbackRate !== 1 ? 'is-active' : ''}`}
              onClick={() => setIsSpeedMenuOpen((prev) => !prev)}
              title="Playback speed ( < / > )"
              aria-label={`Playback speed: ${playbackRate}x`}
            >
              {playbackRate === 1 ? '1x' : `${playbackRate}x`}
            </button>
            {isSpeedMenuOpen && (
              <div className="modal-video-speed-menu" role="menu">
                <div className="modal-video-speed-menu-title">Playback speed</div>
                {PLAYBACK_RATES.map((rate) => (
                  <button
                    key={rate}
                    type="button"
                    className={`modal-video-speed-menu-item ${playbackRate === rate ? 'active' : ''}`}
                    onClick={() => handleSelectSpeed(rate)}
                  >
                    <span className="speed-check">{playbackRate === rate ? '✓' : ''}</span>
                    <span>{rate === 1 ? 'Normal' : `${rate}x`}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
          {onToggleFitMode && (
            <button
              type="button"
              className={`modal-video-icon-button modal-video-fit-button ${fitMode === "cover" ? "active" : ""}`}
              onClick={onToggleFitMode}
              aria-label={fitMode === "contain" ? "Fill frame" : "Fit frame"}
              title={fitMode === "contain" ? "Fill Frame (No black borders) [C]" : "Fit Video [C]"}
            >
              {fitMode === "cover" ? "⤢" : "⇲"}
            </button>
          )}
          {onToggleTheater && (
            <button
              type="button"
              className={`modal-video-icon-button modal-video-theater-button ${isTheaterMode ? "active" : ""}`}
              onClick={onToggleTheater}
              aria-label={isTheaterMode ? "Exit theater mode" : "Expand player"}
              title={isTheaterMode ? "Exit theater mode [T]" : "Expand player [T]"}
            >
              {isTheaterMode ? "⤡" : "⤢"}
            </button>
          )}
          <button
            type="button"
            className="modal-video-icon-button modal-video-fullscreen-button"
            onClick={toggleFullscreen}
            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            title="Fullscreen [F]"
          >
            ⛶
          </button>
        </div>
      </div>
    </div>
  );
};

export default VideoTimeline;
