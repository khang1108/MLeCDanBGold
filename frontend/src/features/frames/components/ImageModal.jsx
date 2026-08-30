import React, { useCallback, useEffect, useMemo, useState } from "react";
import FrameMetadata from "./FrameMetadata";
import VideoTimeline from "./VideoTimeline";
import {
  displayVideoId,
  getStreamVideoUrl,
  normalizeSubmissionFps,
} from "../videoSource";

// The player page endpoint returns HTML, so the inspector uses the raw MP4
// stream and seeks native media time to the selected canonical timestamp.
const ImageModal = ({ frame, query, onSubmit, onClose }) => {
  const modalCardRef = React.useRef(null);
  const videoRef = React.useRef(null);
  const [videoError, setVideoError] = useState(null);
  const [videoDuration, setVideoDuration] = useState(0);
  const targetTime = useMemo(
    () => {
      const timestampMs = Number(frame.timestamp_ms);
      return Number.isInteger(timestampMs) && timestampMs >= 0
        ? timestampMs / 1000
        : null;
    },
    [frame.timestamp_ms],
  );
  const streamUrl = useMemo(
    () => getStreamVideoUrl(frame.video_id, frame.timestamp_ms),
    [frame.timestamp_ms, frame.video_id],
  );
  const [playbackTime, setPlaybackTime] = useState(targetTime);
  const [submitted, setSubmitted] = useState(false);
  const liveFrameIdx = useMemo(() => {
    const submissionFps = normalizeSubmissionFps(frame.fps);
    return Number.isFinite(playbackTime) && submissionFps !== null
      ? Math.round(playbackTime * submissionFps)
      : frame.frame_idx;
  }, [frame.frame_idx, frame.fps, playbackTime]);
  const videoLabel = displayVideoId(frame.video_id);

  useEffect(() => {
    setPlaybackTime(targetTime);
    setVideoError(null);
    setVideoDuration(0);
  }, [streamUrl, targetTime]);

  useEffect(() => {
    const closeOnEscape = (event) => {
      if (event.key === "Escape" && !event.defaultPrevented) {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", closeOnEscape);
    modalCardRef.current?.focus();
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  const updatePlaybackTime = useCallback((sourceTime) => {
    const value = Number(sourceTime);
    if (Number.isFinite(value) && value >= 0) {
      setPlaybackTime(value);
    }
  }, []);

  const handleVideoLoadedMetadata = useCallback((event) => {
    const video = event.currentTarget;
    const duration = Number(video.duration);
    setVideoDuration(Number.isFinite(duration) && duration > 0 ? duration : 0);
    if (targetTime === null) return;

    const seekTime = Number.isFinite(duration) && duration >= 0
      ? Math.min(targetTime, duration)
      : targetTime;
    video.currentTime = seekTime;
    updatePlaybackTime(seekTime);
  }, [targetTime, updatePlaybackTime]);

  const handleVideoTimeUpdate = useCallback((event) => {
    updatePlaybackTime(event.currentTarget.currentTime);
  }, [updatePlaybackTime]);

  const handleVideoSeek = useCallback((nextTime) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(nextTime)) return;
    video.currentTime = nextTime;
    updatePlaybackTime(nextTime);
  }, [updatePlaybackTime]);

  const togglePlayback = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused || video.ended) {
      video.play?.().catch?.(() => undefined);
    } else {
      video.pause?.();
    }
  }, []);

  const seekBy = useCallback((offsetSeconds) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(video.currentTime)) return;
    const duration = Number(video.duration);
    const maximum = Number.isFinite(duration) && duration >= 0 ? duration : Infinity;
    const nextTime = Math.min(Math.max(video.currentTime + offsetSeconds, 0), maximum);
    video.currentTime = nextTime;
    updatePlaybackTime(nextTime);
  }, [updatePlaybackTime]);

  const handleSubmit = useCallback(() => {
    if (!onSubmit) return;
    onSubmit({
      line: `${videoLabel},${liveFrameIdx}`,
      source: "Frame inspector",
    });
    setSubmitted(true);
    window.setTimeout(() => setSubmitted(false), 1200);
  }, [liveFrameIdx, onSubmit, videoLabel]);

  const handleModalKeyDown = useCallback((event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }

    const targetTag = event.target?.tagName;
    if (['BUTTON', 'INPUT', 'TEXTAREA', 'SELECT'].includes(targetTag)) return;

    if (event.key === ' ' || event.key === 'Spacebar' || event.key.toLowerCase() === 'k') {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      seekBy(-5);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      seekBy(5);
    }

  }, [onClose, seekBy, togglePlayback]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-frame-stack" onClick={(event) => event.stopPropagation()}>
        {query?.trim() && (
          <div className="modal-query-context" role="status" aria-label="Current query">
            <p className="modal-query-text">{query.trim()}</p>
          </div>
        )}
        <div
          ref={modalCardRef}
          className="modal-card split-layout"
          onKeyDown={handleModalKeyDown}
          tabIndex={-1}
        >
        <div className="modal-viewer-column">
          {streamUrl && targetTime !== null && !videoError ? (
            <div className="modal-video-shell">
              <video
                ref={videoRef}
                className="modal-viewer-video"
                src={streamUrl}
                autoPlay
                muted
                preload="metadata"
                playsInline
                aria-label={`Video for ${videoLabel}`}
                onLoadedMetadata={handleVideoLoadedMetadata}
                onTimeUpdate={handleVideoTimeUpdate}
                onError={() => setVideoError('The MP4 stream could not be loaded or decoded.')}
              />
              <VideoTimeline
                videoId={frame.video_id}
                videoRef={videoRef}
                currentTime={playbackTime}
                duration={videoDuration}
                onSeek={handleVideoSeek}
                onTogglePlayback={togglePlayback}
              />
            </div>
          ) : (
            <div className="frame-image-placeholder">
              <p>
                Video playback is unavailable. {videoError || (
                  targetTime === null
                    ? 'The backend response is missing timestamp_ms.'
                    : 'The backend response is missing a canonical video_id.'
                )}
              </p>
            </div>
          )}
        </div>
        <div className="modal-inspector-column">
          <div className="inspector-header">
            <span className="inspector-title">
              {videoLabel} · {frame.frame_idx}
            </span>
            <div className="inspector-header-actions">
              {onSubmit && (
                <button
                  type="button"
                  className={`inspector-submit-btn ${submitted ? "submitted" : ""}`}
                  onClick={handleSubmit}
                  aria-label="Submit current frame"
                >
                  {submitted ? "✓" : "Submit"}
                </button>
              )}
              <button
                type="button"
                className="inspector-close-btn"
                onClick={onClose}
                aria-label="Close popup"
              >
                ×
              </button>
            </div>
          </div>
          <div className="inspector-content">
            <div className="inspector-section">
              <span className="inspector-section-label">Caption</span>
              <p className="inspector-caption-text">
                {frame.metadata?.caption ?? frame.caption ?? "No caption available"}
              </p>
            </div>
            <div className="inspector-section">
              <span className="inspector-section-label">Metadata</span>
              <FrameMetadata frame={frame} playbackTime={playbackTime} />
            </div>
          </div>
        </div>
        </div>
      </div>
    </div>
  );
};

export default ImageModal;
