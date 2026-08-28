import React from "react";
import ScoreBreakdown from "./ScoreBreakdown";
import { displayVideoId } from "../videoSource";

// Compact result card; clicking opens the inspector while controls stop propagation.
const FrameCard = ({ frame, onClick, onSubmit }) => {
  const frameId = frame.frame_id;
  const previewUrl = frame.thumbnail_url || frame.frame_url;
  const hasScore = Number.isFinite(frame.scores?.final);
  const hasTimestamp = Number.isFinite(frame.timestamp_ms);
  const submitFrame = (event) => {
    event.stopPropagation();
    onSubmit?.(frame);
  };

  return (
      <div className="frame-card" onClick={onClick}>
      <div className="frame-card-header">
        <span className="frame-index-text">
          {displayVideoId(frame.video_id)}, {frame.frame_idx}
        </span>
        {onSubmit && (
          <div className="frame-card-actions">
            <button
              type="button"
              className="card-submit-btn"
              onClick={submitFrame}
              title="Submit this frame"
              aria-label="Submit this frame"
            >
              Submit
            </button>
          </div>
        )}
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
            alt={`Frame ${frameId}`}
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
      {(hasScore || hasTimestamp) && (
        <div className="frame-card-footer">
          {hasScore && (
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
          )}
          {hasTimestamp && (
            <span className="frame-time-badge">{frame.timestamp_ms} ms</span>
          )}
        </div>
      )}
    </div>
  );
};

export default FrameCard;
