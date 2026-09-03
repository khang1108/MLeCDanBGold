import React from "react";
import { keyframeUrl } from "../../../api/keyframes";
import AlignmentAccordion from "../../alignment/components/AlignmentAccordion";
import { displayVideoId } from "../videoSource";

// Compact result card; clicking opens the inspector while controls stop propagation.
const FrameCard = ({
  frame,
  events = [],
  detail = null,
  detailStatus = 'idle',
  imageLoading = 'lazy',
  className = '',
  onClick,
  onSubmit,
}) => {
  const displayFrame = detail ? { ...frame, ...detail } : frame;
  const frameId = displayFrame.frame_id;
  const previewUrl = frameId ? keyframeUrl(frameId) : null;
  const caption = displayFrame.metadata?.caption ?? displayFrame.caption;
  const hasScore = Number.isFinite(displayFrame.score);
  const hasTimestamp = Number.isFinite(displayFrame.timestamp_ms);
  const submitFrame = (event) => {
    event.stopPropagation();
    onSubmit?.(displayFrame);
  };

  return (
      <div className={`frame-card ${className}`.trim()} onClick={onClick}>
      <div className="frame-card-header">
        <span className="frame-index-text">
          {displayVideoId(displayFrame.video_id)}, {displayFrame.frame_idx}
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
        {previewUrl && detailStatus !== 'loading' ? (
          <img
            src={previewUrl}
            alt={`Frame ${frameId}`}
            className="frame-image"
            loading={imageLoading}
          />
        ) : (
          <div className="frame-image-placeholder">
            {detailStatus === 'loading'
              ? 'Loading frame…'
              : 'Preview unavailable'}
          </div>
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
              Alignment score: {displayFrame.score.toFixed(3)}
            </span>
          )}
          {hasTimestamp && (
            <span className="frame-time-badge">{displayFrame.timestamp_ms} ms</span>
          )}
        </div>
      )}
      <AlignmentAccordion
        events={events}
        frameIds={displayFrame.frame_ids}
        timestampsMs={displayFrame.timestamps_ms}
      />
    </div>
  );
};

export default FrameCard;
