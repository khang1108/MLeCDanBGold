import React from "react";
import { keyframeUrl } from "../../../api/keyframes";
import AlignmentAccordion from "../../alignment/components/AlignmentAccordion";
import { displayVideoId } from "../videoSource";

// Compact result card; clicking opens the inspector while controls stop propagation.
const FrameCard = ({ frame, events = [], onClick, onSubmit }) => {
  const frameId = frame.frame_id;
  const previewUrl = frameId ? keyframeUrl(frameId) : null;
  const caption = frame.metadata?.caption ?? frame.caption;
  const hasScore = Number.isFinite(frame.score);
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
        {caption && (
          <div className="frame-tooltip">
            {caption}
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
        <p className="caption frame-caption-text" title={caption || "No caption available"}>
          {caption || "No caption available"}
        </p>
      </div>
      {(hasScore || hasTimestamp) && (
        <div className="frame-card-footer">
          {hasScore && (
            <span className="frame-score-badge">
              Alignment score: {frame.score.toFixed(3)}
            </span>
          )}
          {hasTimestamp && (
            <span className="frame-time-badge">{frame.timestamp_ms} ms</span>
          )}
        </div>
      )}
      <AlignmentAccordion
        events={events}
        frameIds={frame.frame_ids}
        timestampsMs={frame.timestamps_ms}
      />
    </div>
  );
};

export default FrameCard;
