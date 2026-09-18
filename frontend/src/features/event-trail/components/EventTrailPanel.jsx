import React, { useMemo } from 'react';
import EventRail from './EventRail';
import EvidenceInspector from './EvidenceInspector';
import TrailWindowControls from './TrailWindowControls';
import HypothesisAlternatives from './HypothesisAlternatives';
import HypothesisPathPreview from './HypothesisPathPreview';

const EventTrailPanel = ({
  events = [],
  state = null,
  pending = false,
  error = null,
  selectedEventId = null,
  onSelectEvent,
  onUse,
  onKeep,
  onApprove,
  onRejectMode,
  onClearAnchor,
  onUndo,
  onSetWindow,
  onClearWindow,
  onBack,
  onExitTrail,
  onSubmit,
  // Result-Hypothesis extensions:
  alternatives = [],
  previewAlternative = null,
  isLoadingAlternatives = false,
  currentModeId = null,
  focusedModeId = null,
  currentQueryRevision = null,
  onFocusEvent,
  onPreviewAlternative,
  onClearPreview,
  onUseAlternative,
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

  const effectiveAlternatives = alternatives?.length > 0 ? alternatives : (state.alternatives || []);
  const resolvedModeId =
    currentModeId ||
    focusedModeId ||
    previewAlternative?.alternative_id ||
    previewAlternative?.mode_id ||
    effectiveAlternatives.find((a) => a.is_current)?.alternative_id ||
    effectiveAlternatives.find((a) => a.is_current)?.mode_id ||
    effectiveAlternatives[0]?.alternative_id ||
    effectiveAlternatives[0]?.mode_id ||
    null;

  const handleSelectEvent = (eventId) => {
    onSelectEvent?.(eventId);
    onFocusEvent?.(eventId);
  };

  const handleKeep = (eventId) => {
    if (onKeep) {
      onKeep(eventId);
    } else {
      onApprove?.(eventId);
    }
  };

  const handleReject = (eventId, modeId) => {
    const targetMode = modeId || resolvedModeId;
    onRejectMode?.(eventId, targetMode);
  };

  const handleCycleIndirectEvents = () => {
    const indirectIds = state.transition?.indirect_changed_event_ids;
    if (!Array.isArray(indirectIds) || indirectIds.length === 0) return;
    const currentIndex = indirectIds.indexOf(activeEventId);
    const nextIndex = (currentIndex + 1) % indirectIds.length;
    const nextEventId = indirectIds[nextIndex];
    handleSelectEvent(nextEventId);
  };

  return (
    <section className="event-trail-panel" aria-label="Hypothesis Explorer (EventTrail Exploration)">
      <div className="event-trail-header">
        <div className="event-trail-title-group">
          <h2 className="event-trail-heading">Hypothesis Explorer</h2>
          <span className="event-trail-badge badge-rev">Rev {state.trail_revision}</span>
          {isExhausted ? (
            <span className="event-trail-badge badge-status-exhausted">Exhausted</span>
          ) : (
            <span className="event-trail-badge badge-status-active">Active</span>
          )}
        </div>
        <div className="event-trail-header-actions">
          {(onExitTrail || onBack) && (
            <button
              type="button"
              className="btn-danger btn-sm event-trail-exit-btn"
              onClick={() => (onExitTrail ? onExitTrail() : onBack?.())}
              title="Exit EventTrail and return to Frame Inspector"
            >
              Exit
            </button>
          )}
        </div>
      </div>

      {Number.isInteger(currentQueryRevision) &&
        Number.isInteger(state.kis_revision) &&
        state.kis_revision !== currentQueryRevision && (
          <div
            className="event-trail-stale-query-banner"
            role="status"
            data-testid="trail-stale-query-notice"
          >
            Based on query revision {state.kis_revision}
          </div>
        )}

      {error && (
        <div className="event-trail-error-banner" role="alert">
          {error}
        </div>
      )}

      {isExhausted && (
        <div className="event-trail-exhausted-banner" role="alert">
          <strong>No valid path remains in this video.</strong> Use Undo or Exit to recover.
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
        onSelectEvent={handleSelectEvent}
      />

      <HypothesisAlternatives
        alternatives={effectiveAlternatives}
        activeAlternativeId={previewAlternative?.alternative_id || previewAlternative?.mode_id || null}
        isLoading={isLoadingAlternatives}
        disabled={isExhausted || pending}
        onPreview={onPreviewAlternative}
        onUse={onUseAlternative}
      />

      {previewAlternative && (
        <HypothesisPathPreview
          activePath={candidates}
          previewAlternative={previewAlternative}
          pending={pending}
          onUseAlternative={onUseAlternative}
          onClearPreview={onClearPreview}
        />
      )}

      <EvidenceInspector
        selectedEventId={activeEventId}
        event={selectedEvent}
        candidate={selectedCandidate}
        isApproved={isApproved}
        isExhausted={isExhausted}
        pending={pending}
        rejectedCount={rejectedCount}
        diff={currentDiff}
        currentModeId={resolvedModeId}
        previewAlternative={previewAlternative}
        onKeep={handleKeep}
        onApprove={handleKeep}
        onRejectMode={handleReject}
        onUse={onUse}
        onUseAlternative={onUseAlternative}
        onClearAnchor={onClearAnchor}
        onClearPreview={onClearPreview}
      />

      <TrailWindowControls
        window={state.window}
        pending={pending}
        onSetWindow={onSetWindow}
        onClearWindow={onClearWindow}
      />

      <div className="event-trail-footer">
        {!state.submission_selection && !isExhausted && (
          <p className="event-trail-submit-hint">
            Use a frame before submitting from EventTrail.
          </p>
        )}
        <div className="event-trail-footer-actions">
          <button
            type="button"
            className="btn-secondary event-trail-footer-btn undo-btn"
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
      </div>
    </section>
  );
};

export default EventTrailPanel;
