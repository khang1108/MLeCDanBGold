import React, { useMemo } from 'react';
import EventRail from './EventRail';
import EvidenceInspector from './EvidenceInspector';
import TrailWindowControls from './TrailWindowControls';

const EventTrailPanel = ({
  events = [],
  state = null,
  pending = false,
  error = null,
  selectedEventId = null,
  onSelectEvent,
  onExplore,
  onUse,
  onApprove,
  onDecline,
  onClearAnchor,
  onUndo,
  onSetWindow,
  onClearWindow,
  onBack,
  onSubmit,
}) => {
  const activeEventId = useMemo(() => {
    if (selectedEventId) return selectedEventId;
    return events.length > 0 ? events[0].id : null;
  }, [selectedEventId, events]);

  if (!state) return null;

  const isExhausted = state.status === 'exhausted';
  const candidates = isExhausted ? (state.last_valid_path || []) : (state.path || []);
  const selectedCandidate = candidates.find((c) => c.event_id === activeEventId);
  const selectedEvent = events.find((e) => e.id === activeEventId);
  const isApproved = Boolean(state.approved_event_ids?.includes(activeEventId));
  const rejectedCount = state.rejected_counts?.[activeEventId] || 0;
  const currentDiff = state.transition?.candidate_diffs?.find((d) => d.event_id === activeEventId) || null;

  const indirectCount = state.transition?.indirect_changed_event_ids?.length || 0;
  const canSubmit = !isExhausted && Boolean(state.submission_selection) && !pending;

  const handleCycleIndirectEvents = () => {
    const indirectIds = state.transition?.indirect_changed_event_ids;
    if (!Array.isArray(indirectIds) || indirectIds.length === 0) return;
    const currentIndex = indirectIds.indexOf(activeEventId);
    const nextIndex = (currentIndex + 1) % indirectIds.length;
    const nextEventId = indirectIds[nextIndex];
    onSelectEvent?.(nextEventId);
  };

  return (
    <section className="event-trail-panel" aria-label="EventTrail Exploration">
      <div className="event-trail-header">
        <div className="event-trail-title-group">
          <h2 className="event-trail-heading">EventTrail</h2>
          <span className="event-trail-badge badge-rev">Rev {state.trail_revision}</span>
          {isExhausted ? (
            <span className="event-trail-badge badge-status-exhausted">Exhausted</span>
          ) : (
            <span className="event-trail-badge badge-status-active">Active</span>
          )}
        </div>
        {onBack && (
          <button
            type="button"
            className="btn-secondary btn-sm event-trail-back-btn"
            onClick={onBack}
          >
            Back to results
          </button>
        )}
      </div>

      {error && (
        <div className="event-trail-error-banner" role="alert">
          {error}
        </div>
      )}

      {isExhausted && (
        <div className="event-trail-exhausted-banner" role="alert">
          <strong>No valid path remains in this video.</strong> Use Undo or Back to results to recover.
        </div>
      )}

      {state.transition && !isExhausted && (
        <div className="event-trail-transition-banner" role="status">
          <span className="transition-icon">⚡</span>
          <span>
            {state.transition.action_event_id
              ? `Action applied to ${state.transition.action_event_id}`
              : 'Window range applied'}
          </span>
          {indirectCount > 0 && (
            <button
              type="button"
              className="event-trail-cycle-btn"
              onClick={handleCycleIndirectEvents}
              title="Cycle through updated events"
            >
              {indirectCount} other events updated
            </button>
          )}
        </div>
      )}

      <EventRail
        events={events}
        path={state.path}
        lastValidPath={state.last_valid_path}
        isExhausted={isExhausted}
        approvedEventIds={state.approved_event_ids}
        rejectedCounts={state.rejected_counts}
        selectedEventId={activeEventId}
        transition={state.transition}
        onSelectEvent={onSelectEvent}
        onExplore={onExplore}
      />

      <EvidenceInspector
        selectedEventId={activeEventId}
        event={selectedEvent}
        candidate={selectedCandidate}
        isApproved={isApproved}
        isExhausted={isExhausted}
        pending={pending}
        rejectedCount={rejectedCount}
        diff={currentDiff}
        onApprove={onApprove}
        onDecline={onDecline}
        onUse={onUse}
        onClearAnchor={onClearAnchor}
      />

      <TrailWindowControls
        window={state.window}
        pending={pending}
        onSetWindow={onSetWindow}
        onClearWindow={onClearWindow}
      />

      <div className="event-trail-footer-actions">
        {!state.submission_selection && !isExhausted && (
          <p className="event-trail-submit-hint">Use a frame before submitting from EventTrail.</p>
        )}
        <button
          type="button"
          className="btn-secondary event-trail-footer-btn"
          disabled={pending}
          onClick={onUndo}
        >
          Undo
        </button>

        <button
          type="button"
          className="btn-primary event-trail-footer-btn submit-btn"
          disabled={!canSubmit}
          onClick={() => onSubmit?.(state.submission_selection)}
        >
          Submit
        </button>
      </div>
    </section>
  );
};

export default EventTrailPanel;
