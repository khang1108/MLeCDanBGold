import React, { useState } from "react";
import FrameFeedbackActions from "./FrameFeedbackActions";
import ScoreBreakdown from "./ScoreBreakdown";

// Compact result card; clicking opens the inspector while controls stop propagation.
const FrameCard = ({
  frame,
  feedbackState,
  onClick,
  onPromising,
  onReject,
  onChallengeSubmit,
  isChallengeSubmitting,
}) => {
  const [copied, setCopied] = useState(false);
  const previewUrl = frame.thumbnail_url || frame.frame_url;
  const copy = (event) => {
    event.stopPropagation();
    navigator.clipboard.writeText(`${frame.video_id},${frame.frame_idx}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  return (
    <div
      className={`frame-card ${feedbackState ? `is-${feedbackState}` : ""}`}
      onClick={onClick}
    >
      <div className="frame-tooltip">
        {frame.caption}
        <div className="frame-tooltip-arrow" />
      </div>
      <div className="frame-card-header">
        <span className="frame-index-text">
          {frame.video_id} · frame {frame.frame_idx}
        </span>
        <div className="frame-card-actions">
          {onChallengeSubmit && (
            <button
              className="card-submit-btn"
              onClick={(event) => {
                event.stopPropagation();
                onChallengeSubmit(frame);
              }}
              disabled={isChallengeSubmitting}
              title="Submit this video to the current mini-challenge task"
            >
              {isChallengeSubmitting ? "Sending…" : "Submit"}
            </button>
          )}
          <button
            className={`card-copy-btn ${copied ? "copied" : ""}`}
            onClick={copy}
            title="Copy official video_id,frame_idx"
          >
            {copied ? "✓" : "⧉"}
          </button>
        </div>
      </div>
      {onPromising && onReject && (
        <FrameFeedbackActions
          state={feedbackState}
          onPromising={onPromising}
          onReject={onReject}
          frameId={frame.frame_id}
        />
      )}
      <div className="frame-image-container">
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
        <p className="caption frame-caption-text">
          {frame.caption || "No caption available"}
        </p>
      </div>
      <div className="frame-card-footer">
        <div className="frame-score-badge-wrapper">
          <span className="frame-score-badge">
            Score: {Math.round(frame.scores.final * 100)}%
          </span>
          <div className="score-tooltip">
            <div className="score-tooltip-title">Score Details</div>
            <ScoreBreakdown scores={frame.scores} />
            <div className="score-tooltip-arrow" />
          </div>
        </div>
        <span className="frame-time-badge">{frame.timestamp_ms} ms</span>
      </div>
    </div>
  );
};

export default FrameCard;
