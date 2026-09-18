import React from 'react';
import { keyframeUrl } from '../../../api/keyframes';

const formatSeconds = (timestampMs) => {
  if (typeof timestampMs !== 'number' || !Number.isFinite(timestampMs)) return '—';
  return `${(timestampMs / 1000).toFixed(2)}s`;
};

const EvidenceInspector = ({
  selectedEventId,
  event,
  candidate,
  isApproved = false,
  isExhausted = false,
  pending = false,
  rejectedCount = 0,
  diff = null,
  currentModeId = null,
  previewAlternative = null,
  onKeep,
  onApprove,
  onRejectMode,
  onDecline,
  onUse,
  onUseAlternative,
  onClearAnchor,
  onClearPreview,
}) => {
  if (!selectedEventId) {
    return (
      <div className="event-trail-inspector empty">
        <p className="body-sm text-muted">Select an event from the rail above to inspect evidence.</p>
      </div>
    );
  }

  const previewUrl = candidate?.frame_id ? keyframeUrl(candidate.frame_id) : null;

  return (
    <div className="event-trail-inspector">
      <div className="event-trail-inspector-header">
        <div className="event-trail-inspector-title">
          <span className="event-trail-event-tag large">{selectedEventId}</span>
          <span className="event-trail-event-desc">{event?.text || '—'}</span>
        </div>
        {isApproved && (
          <span className="event-trail-badge badge-approved">Anchor Locked</span>
        )}
      </div>

      {candidate ? (
        <div className="event-trail-inspector-card">
          <div className="event-trail-inspector-preview">
            {previewUrl ? (
              <img
                src={previewUrl}
                alt={`Selected frame ${candidate.frame_id}`}
                className="event-trail-inspector-img"
              />
            ) : (
              <div className="event-trail-thumb-placeholder">No Frame</div>
            )}
          </div>
          <div className="event-trail-inspector-details">
            <div className="event-trail-meta-row">
              <span className="meta-label">Frame:</span>
              <span className="meta-value">#{candidate.frame_idx}</span>
            </div>
            <div className="event-trail-meta-row">
              <span className="meta-label">Timestamp:</span>
              <span className="meta-value">{formatSeconds(candidate.timestamp_ms)}</span>
            </div>
            {rejectedCount > 0 && (
              <div className="event-trail-meta-row">
                <span className="meta-label">Rejected:</span>
                <span className="meta-value text-warning">{rejectedCount}</span>
              </div>
            )}
            {diff && (
              <div className="event-trail-diff-box">
                <span className="diff-label">Latest Transition:</span>
                <span className="diff-transition">
                  {formatSeconds(diff.before_timestamp_ms)} → {formatSeconds(diff.after_timestamp_ms)}
                </span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <p className="body-sm text-muted">No candidate frame currently assigned.</p>
      )}

      <div className="event-trail-actions-row">
        {previewAlternative && (
          <>
            <button
              type="button"
              className="btn-primary event-trail-action-btn use-alt-btn"
              disabled={isExhausted || pending}
              onClick={() =>
                onUseAlternative?.(
                  selectedEventId,
                  previewAlternative.alternative_id || previewAlternative.mode_id
                )
              }
              title="Commit this complete-path alternative"
            >
              Use this occurrence
            </button>
            <button
              type="button"
              className="btn-secondary event-trail-action-btn clear-preview-btn"
              disabled={pending}
              onClick={() => onClearPreview?.()}
              title="Clear path preview"
            >
              Clear preview
            </button>
          </>
        )}

        <button
          type="button"
          className="btn-primary event-trail-action-btn keep-btn approve-btn"
          disabled={isApproved || isExhausted || pending}
          onClick={() => (onKeep ? onKeep(selectedEventId) : onApprove?.(selectedEventId))}
          title="Keep this candidate occurrence as an anchor"
        >
          Keep
        </button>

        <button
          type="button"
          className="btn-secondary event-trail-action-btn reject-btn decline-btn"
          disabled={isApproved || isExhausted || pending}
          onClick={() =>
            onRejectMode
              ? onRejectMode(selectedEventId, currentModeId)
              : onDecline?.(selectedEventId)
          }
          title="Reject this candidate occurrence to explore other temporal modes"
        >
          Reject occurrence
        </button>

        <button
          type="button"
          className="btn-secondary event-trail-action-btn use-btn"
          disabled={isExhausted || pending}
          onClick={() => onUse?.(selectedEventId)}
          title="Use current player time as canonical frame for this event"
        >
          Use (manual frame)
        </button>

        {isApproved && (
          <button
            type="button"
            className="btn-danger event-trail-action-btn clear-anchor-btn"
            disabled={pending}
            onClick={() => onClearAnchor?.(selectedEventId)}
            title="Clear anchor constraint on this event"
          >
            Clear anchor
          </button>
        )}
      </div>
    </div>
  );
};

export default EvidenceInspector;
