import React, { useState } from "react";
import ScoreBreakdown from "./ScoreBreakdown";
import { displayVideoId } from "../videoSource";

// Compact result card; clicking opens the inspector while controls stop propagation.
const FrameCard = ({
  frame,
  feedbackState,
  onClick,
}) => {
  const [copied, setCopied] = useState(false);
  const previewUrl = frame.thumbnail_url || frame.frame_url;
  const hasScore = Number.isFinite(frame.scores?.final);
  const hasTimestamp = Number.isFinite(frame.timestamp_ms);
  const copy = (event) => {
    event.stopPropagation();
    navigator.clipboard.writeText(`${frame.video_id},${frame.frame_idx}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  return (
    <div
      className="frame-card"
      onClick={onClick}
    >
      <div className="frame-card-header">
        <span className="frame-index-text">
          {displayVideoId(frame.video_id)} &middot; frame {frame.frame_idx}
        </span>
        <div className="frame-card-actions">
          <button
            className={`card-copy-btn ${copied ? "copied" : ""}`}
            onClick={copy}
            title="Copy official video_id,frame_idx"
          >
            {copied ? "✓" : "⧉"}
          </button>
        </div>
      </div>
      <div className="frame-image-container">
        {frame.caption && (
          <div className="frame-tooltip">
            {frame.caption}
            <div className="frame-tooltip-arrow" />
          </div>
        )}
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={`Frame ${frame.frame_id}`}
            className="frame-image"
            loading="lazy"
          />
        ) : (
          <div className="frame-image-placeholder">Preview unavailable</div>
        )}
      </div>
      <div className="frame-caption-container">
        <p className="caption frame-caption-text" title={frame.caption || "No caption available"}>
          {frame.caption || "No caption available"}
        </p>
      </div>
      {frame.answer && (
        <div className="frame-answer-container" title={frame.answer}>
          <p className="caption frame-answer-text vqa-answer-highlight" style={{ fontWeight: '600', color: 'var(--color-primary-light)' }}>
            {frame.answer}
          </p>
        </div>
      )}
      {(hasScore || hasTimestamp) && <div className="frame-card-footer">
        {hasScore && <div className="frame-score-badge-wrapper">
          <span className="frame-score-badge">
            Score: {Math.round(frame.scores.final * 100)}%
          </span>
          <div className="score-tooltip">
            <div className="score-tooltip-title">Score Details</div>
            <ScoreBreakdown scores={frame.scores} />
            <div className="score-tooltip-arrow" />
          </div>
        </div>}
        {hasTimestamp && <span className="frame-time-badge">{frame.timestamp_ms} ms</span>}
      </div>}
    </div>
  );
};

export default FrameCard;
