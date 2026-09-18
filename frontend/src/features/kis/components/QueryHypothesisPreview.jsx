import React from 'react';

/**
 * Preview panel for proposed query hypothesis structural mutations.
 * Exposes only two canonical terminal actions: Cancel and Apply.
 */
const QueryHypothesisPreview = ({
  preview,
  currentIntent,
  onApply,
  onCancel,
  disabled = false,
}) => {
  if (!preview?.intent) return null;

  const proposedEvents = preview.intent.events || [];
  const currentEvents = currentIntent?.events || [];

  return (
    <div className="query-hypothesis-preview" data-testid="query-hypothesis-preview">
      <div className="query-hypothesis-preview-header">
        <h4>Proposed Query Changes</h4>
        <span className="query-hypothesis-revision-diff">
          Rev {preview.base_revision ?? currentIntent?.revision ?? 0} → {preview.intent.revision}
        </span>
      </div>

      <div className="query-hypothesis-preview-comparison">
        <div className="query-hypothesis-preview-side current">
          <span className="side-title">Current</span>
          <ol className="preview-events-list">
            {currentEvents.map((event) => (
              <li key={event.id} className="preview-event-row">
                <span className="event-tag">{event.id}</span>
                <span className="event-body">{event.text}</span>
              </li>
            ))}
          </ol>
        </div>

        <div className="query-hypothesis-preview-side proposed">
          <span className="side-title">Proposed</span>
          <ol className="preview-events-list">
            {proposedEvents.map((event) => (
              <li key={event.id} className="preview-event-row">
                <span className="event-tag">{event.id}</span>
                <span className="event-body">{event.text}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>

      <div className="query-hypothesis-preview-actions">
        <button
          type="button"
          className="btn btn-secondary query-preview-cancel-btn"
          onClick={onCancel}
          disabled={disabled}
        >
          Cancel
        </button>
        <button
          type="button"
          className="btn btn-primary query-preview-apply-btn"
          onClick={onApply}
          disabled={disabled}
        >
          Apply
        </button>
      </div>
    </div>
  );
};

export default QueryHypothesisPreview;
