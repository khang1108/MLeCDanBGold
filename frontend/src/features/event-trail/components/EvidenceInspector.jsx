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
  onApprove,
  onDecline,
  onUse,
  onClearAnchor,
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
              <span className="meta-label">Frame Coordinate:</span>
              <span className="meta-value">#{candidate.frame_idx}</span>
            </div>
            <div className="event-trail-meta-row">
              <span className="meta-label">Timestamp:</span>
              <span className="meta-value">{formatSeconds(candidate.timestamp_ms)} ({candidate.timestamp_ms} ms)</span>
            </div>
            {rejectedCount > 0 && (
              <div className="event-trail-meta-row">
                <span className="meta-label">Declined candidates:</span>
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
        <button
          type="button"
          className="btn-primary event-trail-action-btn approve-btn"
          disabled={isApproved || isExhausted || pending}
          onClick={() => onApprove?.(selectedEventId)}
          title="Approve this candidate as an anchor"
        >
          Approve
        </button>

        <button
          type="button"
          className="btn-secondary event-trail-action-btn decline-btn"
          disabled={isApproved || isExhausted || pending}
          onClick={() => onDecline?.(selectedEventId)}
          title="Decline this candidate to search elsewhere in this video"
        >
          Decline
        </button>

        <button
          type="button"
          className="btn-secondary event-trail-action-btn use-btn"
          disabled={isExhausted || pending}
          onClick={() => onUse?.(selectedEventId)}
          title="Use current player time as canonical frame for this event"
        >
          Use
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
