import React, { useCallback, useEffect, useMemo, useState } from "react";
import FrameMetadata from "./FrameMetadata";
import ScoreBreakdown from "./ScoreBreakdown";
import YouTubePlayer from "./YouTubePlayer";
import {
  displayVideoId,
  getYouTubeEmbedUrl,
  getYouTubeWatchUrl,
  timestampSeconds,
} from "../videoSource";

// Inspector uses the official YouTube player and seeks by canonical timestamp.
// BTC frame_idx remains the submission coordinate; timestamp_ms controls playback.
const ImageModal = ({ frame, onClose }) => {
  const modalCardRef = React.useRef(null);
  const [copied, setCopied] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [videoError, setVideoError] = useState(null);
  const playerControlsRef = React.useRef(null);
  const targetTime = useMemo(
    () => timestampSeconds(frame.timestamp_ms),
    [frame.timestamp_ms],
  );
  const [playbackTime, setPlaybackTime] = useState(targetTime);
  const videoLabel = displayVideoId(frame.video_id);
  const watchUrl = getYouTubeWatchUrl(frame.video_id);

  useEffect(() => {
    setPlaybackTime(targetTime);
  }, [targetTime]);

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

  useEffect(() => {
    const embedUrl = getYouTubeEmbedUrl(frame.video_id);
    setVideoUrl(embedUrl);
    setVideoError(embedUrl ? null : 'No YouTube URL is mapped for this video.');
  }, [frame.video_id]);

  const updatePlaybackTime = useCallback((currentTime) => {
    const value = Number(currentTime);
    if (Number.isFinite(value) && value >= 0) setPlaybackTime(value);
  }, []);

  const keepModalFocused = useCallback(() => {
    // The API-created YouTube iframe is cross-origin. Move focus back after
    // the mouse action so Escape and modal keyboard shortcuts remain usable.
    window.setTimeout(() => modalCardRef.current?.focus(), 50);
  }, []);

  const handlePlayerFocus = useCallback(() => {
    keepModalFocused();
  }, [keepModalFocused]);

  const handlePlayerReady = useCallback((controls) => {
    playerControlsRef.current = controls;
  }, []);

  const handleModalKeyDown = useCallback((event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }

    const controls = playerControlsRef.current;
    if (!controls) return;

    const targetTag = event.target?.tagName;
    if (targetTag && ['BUTTON', 'A', 'INPUT', 'TEXTAREA', 'SELECT'].includes(targetTag)) return;

    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      controls.seekBy(event.key === 'ArrowLeft' ? -5 : 5);
    } else if (event.key === ' ' || event.key.toLowerCase() === 'k') {
      event.preventDefault();
      controls.togglePlayPause();
    }
  }, [onClose]);

  const copy = () => {
    navigator.clipboard.writeText(`${frame.video_id},${frame.frame_idx}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        ref={modalCardRef}
        className="modal-card split-layout"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={handleModalKeyDown}
        tabIndex={-1}
      >
        <div className="modal-viewer-column">
          {videoUrl && targetTime !== null && !videoError ? (
            <div className="modal-video-shell">
              <YouTubePlayer
                embedUrl={videoUrl}
                targetTime={targetTime}
                title={`Video for ${videoLabel}`}
                onTimeUpdate={updatePlaybackTime}
                onIframeFocus={handlePlayerFocus}
                onPlayerReady={handlePlayerReady}
              />
            </div>
          ) : videoError || targetTime === null ? (
            <div className="frame-image-placeholder">
              <p>
                Video playback is unavailable. {videoError || 'The backend response is missing timestamp_ms.'}
              </p>
              {watchUrl && (
                <a
                  className="youtube-fallback-link"
                  href={watchUrl}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open on YouTube
                </a>
              )}
            </div>
          ) : (
            <div className="frame-image-placeholder">Preparing YouTube playback…</div>
          )}
        </div>
        <div className="modal-inspector-column">
          <div className="inspector-header">
            <span className="inspector-title">
              {videoLabel} · BTC frame {frame.frame_idx}
            </span>
            <div className="inspector-header-actions">
              <button
                className={`inspector-copy-btn ${copied ? "copied" : ""}`}
                onClick={copy}
                title="Copy official video_id,frame_idx"
              >
                {copied ? "✓" : "⧉"}
              </button>
              <button
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
                {frame.caption || "No caption available"}
              </p>
            </div>
            {frame.answer && (
              <div className="inspector-section">
                <span className="inspector-section-label">VQA Answer</span>
                <p className="inspector-caption-text vqa-answer-highlight" style={{ fontWeight: '600', color: 'var(--color-primary-light)' }}>
                  {frame.answer}
                </p>
              </div>
            )}
            <div className="inspector-section">
              <span className="inspector-section-label">Metadata</span>
              <FrameMetadata frame={frame} playbackTime={playbackTime} />
            </div>
            {Number.isFinite(frame.scores?.final) && <div className="inspector-section">
              <span className="inspector-section-label">
                Retrieval Stage Scores
              </span>
              <div className="inspector-scores-grid">
                <ScoreBreakdown scores={frame.scores} asRows />
              </div>
            </div>}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageModal;
