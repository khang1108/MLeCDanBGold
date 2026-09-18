import React from "react";
import { keyframeUrl } from "../../../api/keyframes";
import { displayVideoId } from "../videoSource";

// Compact result card; clicking opens the inspector while controls stop propagation.
const FrameCard = ({
  frame,
  eventLabel = null,
  events = [],
  detail = null,
  detailStatus = 'idle',
  imageLoading = 'lazy',
  className = '',
  annotation = 'unvisited',
  onOpenSubmission,
  isSubmissionOpening = false,
  onClick,
  onSeek,
}) => {
  const displayFrame = detail ? { ...frame, ...detail } : frame;
  const frameId = displayFrame.frame_id;
  const cardClassName = [
    className,
    annotation && annotation !== 'unvisited' ? `trail-${annotation}` : '',
  ].filter(Boolean).join(' ');
  const previewUrl = frameId ? keyframeUrl(frameId) : null;
  const hasTimestamp = Number.isFinite(displayFrame.timestamp_ms);
  const canSubmitFrame = typeof onOpenSubmission === 'function'
    && typeof frame.video_id === 'string'
    && frame.video_id.trim().length > 0
    && Number.isSafeInteger(frame.timestamp_ms)
    && frame.timestamp_ms >= 0;
  return (
    <div className={`frame-card ${cardClassName}`.trim()} onClick={onClick}>
      <div className="frame-card-header">
        <div className="frame-header-title">
          {eventLabel && (
            <span className="frame-event-badge">{eventLabel}</span>
          )}
          <span className="frame-index-text" title={displayFrame.video_id}>
            {displayVideoId(displayFrame.video_id)}
          </span>
        </div>
        <div className="frame-header-meta">
          {annotation === 'explored' && (
            <span className="frame-trail-badge badge-explored">Explored</span>
          )}
          {annotation === 'exhausted' && (
            <span className="frame-trail-badge badge-exhausted">Exhausted</span>
          )}
          {hasTimestamp && (
            <span className="frame-time-badge">{displayFrame.timestamp_ms} ms</span>
          )}
          {canSubmitFrame && (
            <button
              type="button"
              className="frame-submit-button"
              disabled={isSubmissionOpening}
              onClick={(event) => {
                event.stopPropagation();
                onOpenSubmission?.({
                  videoId: frame.video_id,
                  startMs: frame.timestamp_ms,
                  endMs: frame.timestamp_ms,
                });
              }}
              aria-label="Submit this frame to DRES"
              title="Submit this frame to DRES"
            >
              ↗
            </button>
          )}
        </div>
      </div>
      <div className="frame-image-container">
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
    </div>
  );
};

export default FrameCard;
