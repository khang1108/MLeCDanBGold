import React from 'react';
import { keyframeUrl } from '../../../api/keyframes';

const formatSeconds = (timestampMs) => {
  if (typeof timestampMs !== 'number' || !Number.isFinite(timestampMs)) return '—';
  return `${(timestampMs / 1000).toFixed(2)}s`;
};

const EventRail = ({
  events = [],
  path = [],
  lastValidPath = [],
  isExhausted = false,
  approvedEventIds = [],
  rejectedCounts = {},
  selectedEventId = null,
  transition = null,
  onSelectEvent,
  onExplore,
}) => {
  const candidates = isExhausted ? (lastValidPath || []) : (path || []);

  return (
    <div className={`event-trail-rail ${isExhausted ? 'event-trail-rail-dimmed' : ''}`}>
      <div className="event-trail-rail-header">
        <span className="event-trail-section-label">Temporal Event Sequence</span>
        {isExhausted && (
          <span className="event-trail-badge badge-exhausted">Last Valid Path</span>
        )}
      </div>
      <div className="event-trail-rail-list">
        {events.map((event) => {
          const candidate = candidates.find((c) => c.event_id === event.id);
          const isApproved = approvedEventIds.includes(event.id);
          const isSelected = selectedEventId === event.id;
          const isDirect = transition?.direct_changed_event_ids?.includes(event.id);
          const isIndirect = transition?.indirect_changed_event_ids?.includes(event.id);
          const rejectedCount = rejectedCounts?.[event.id] || 0;
          const previewUrl = candidate?.frame_id ? keyframeUrl(candidate.frame_id) : null;

          return (
            <div
              key={event.id}
              data-testid={`event-rail-item-${event.id}`}
              className={`event-trail-rail-item ${isSelected ? 'selected' : ''} ${isApproved ? 'approved' : ''}`}
              onClick={() => onSelectEvent?.(event.id)}
            >
              <div className="event-trail-item-top">
                <span className="event-trail-event-tag">{event.id}</span>
                <span className="event-trail-event-text" title={event.text}>
                  {event.text}
                </span>
                <div className="event-trail-item-badges">
                  {isApproved && (
                    <span className="event-trail-badge badge-approved" title="Anchored frame">
                      ⚓ Anchor
                    </span>
                  )}
                  {!isApproved && rejectedCount > 0 && (
                    <span className="event-trail-badge badge-declined" title={`${rejectedCount} declined candidates`}>
                      ✕ {rejectedCount}
                    </span>
                  )}
                  {isDirect && (
                    <span className="event-trail-badge badge-direct">Direct</span>
                  )}
                  {!isDirect && isIndirect && (
                    <span className="event-trail-badge badge-indirect">Updated</span>
                  )}
                </div>
              </div>

              {candidate && (
                <div className="event-trail-candidate-card">
                  <div className="event-trail-thumb-wrapper">
                    {previewUrl ? (
                      <img
                        src={previewUrl}
                        alt={`Frame ${candidate.frame_id}`}
                        className="event-trail-thumb"
                        loading="lazy"
                      />
                    ) : (
                      <div className="event-trail-thumb-placeholder">—</div>
                    )}
                  </div>
                  <div className="event-trail-candidate-meta">
                    <span className="event-trail-coord">
                      #{candidate.frame_idx} · {formatSeconds(candidate.timestamp_ms)}
                    </span>
                    <button
                      type="button"
                      className="btn-secondary btn-sm event-trail-explore-btn"
                      aria-label={`Explore ${event.id}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onExplore?.(candidate);
                      }}
                    >
                      Explore
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default EventRail;
