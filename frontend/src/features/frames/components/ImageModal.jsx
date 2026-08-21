import React, { useCallback, useEffect, useMemo, useState } from "react";
import FrameMetadata from "./FrameMetadata";
import ScoreBreakdown from "./ScoreBreakdown";
import {
  displayVideoId,
  frameIndexAt,
  getS3VideoUrl,
  targetTimeSeconds,
} from "../videoSource";

// Inspector streams the source MP4 directly from S3 and seeks to the selected frame.
const ImageModal = ({ frame, onClose }) => {
  const [copied, setCopied] = useState(false);
  const [currentFrameIdx, setCurrentFrameIdx] = useState(frame.frame_idx);
  const [videoUrl, setVideoUrl] = useState(null);
  const [videoError, setVideoError] = useState(null);
  const targetTime = useMemo(
    () => targetTimeSeconds(frame.frame_idx, frame.fps),
    [frame.frame_idx, frame.fps],
  );
  const videoLabel = displayVideoId(frame.video_id);

  useEffect(() => {
    const closeOnEscape = (event) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  useEffect(() => {
    setCurrentFrameIdx(frame.frame_idx);
  }, [frame.frame_idx, frame.fps, videoUrl]);

  useEffect(() => {
    let active = true;
    setVideoUrl(null);
    setVideoError(null);
    getS3VideoUrl(frame.video_id)
      .then((url) => {
        if (!active) return;
        if (url) setVideoUrl(url);
        else setVideoError('Configure S3 bucket, region, and AWS credentials, and ensure the backend returns fps.');
      })
      .catch(() => {
        if (active) setVideoError('Could not create a temporary S3 video URL. Check your local AWS configuration.');
      });
    return () => {
      active = false;
    };
  }, [frame.video_id]);

  const updateCurrentFrame = useCallback((currentTime) => {
    const nextFrameIdx = frameIndexAt(currentTime, frame.fps);
    if (nextFrameIdx !== null) setCurrentFrameIdx(nextFrameIdx);
  }, [frame.fps]);

  const seekToTarget = useCallback((event) => {
    if (targetTime === null) return;
    const video = event.currentTarget;
    const duration = Number(video.duration);
    const seekTime = Number.isFinite(duration)
      ? Math.min(targetTime, Math.max(0, duration))
      : targetTime;
    video.currentTime = seekTime;
    updateCurrentFrame(seekTime);
  }, [targetTime, updateCurrentFrame]);

  const copy = () => {
    navigator.clipboard.writeText(`${frame.video_id},${frame.frame_idx}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-card split-layout"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-viewer-column">
          {videoUrl && targetTime !== null ? (
            <div className="modal-video-shell">
              <video
                aria-label={`Video for ${videoLabel}`}
                className="modal-viewer-video"
                controls
                playsInline
                preload="metadata"
                src={videoUrl}
                onLoadedMetadata={seekToTarget}
                onTimeUpdate={(event) => updateCurrentFrame(event.currentTarget.currentTime)}
              />
              <output className="modal-video-frame-badge" aria-live="off">
                Frame {currentFrameIdx}
              </output>
            </div>
          ) : videoError || targetTime === null ? (
            <div className="frame-image-placeholder">
              Video playback is unavailable. {videoError || 'Ensure the backend returns a valid fps value.'}
            </div>
          ) : (
            <div className="frame-image-placeholder">Preparing secure video playback…</div>
          )}
        </div>
        <div className="modal-inspector-column">
          <div className="inspector-header">
            <span className="inspector-title">
              {videoLabel} · frame {frame.frame_idx}
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
              <FrameMetadata frame={frame} />
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
