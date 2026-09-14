import React from 'react';

/**
 * Present one revisioned KIS session as clue history and resolved metadata.
 *
 * This component owns the sole text input for KIS. It intentionally presents
 * retrieval state rather than simulating a chat participant.
 */
const KisPanel = ({
  sessionState = {},
  onDraftChange,
  onSubmit,
  onReset,
  disabled = false,
  inputRef,
  onFocusQueryInput,
  onBlurQueryInput,
  renderExtraActions,
  submitLabel = 'Search',
  resetLabel = 'New Search',
}) => {
  const {
    draft = '',
    committedInputs = [],
    revision = 0,
    currentIntent = null,
    isSearching = false,
    error = null,
  } = sessionState;

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!isSearching && draft.trim()) {
        onSubmit?.(event);
      }
    }
  };

  return (
    <section className="kis-chat-panel" data-testid="kis-panel" aria-label="KIS search">
      <div className="kis-chat-header">
        <div className="kis-chat-header-title">
          <span className="kis-chat-title-text">KIS Semantic Search</span>
          {revision > 0 && (
            <span className="kis-revision-tag" title={`Revision ${revision}`}>
              Rev {revision}
            </span>
          )}
        </div>
        {onReset && (
          <button
            type="button"
            className="kis-chat-reset-btn"
            onClick={onReset}
            title="Start a new KIS search"
          >
            {resetLabel}
          </button>
        )}
      </div>

      <div className="kis-chat-body">
        {committedInputs.length > 0 && (
          <section className="kis-clue-history" aria-label="Committed clues">
            <span className="kis-intent-sublabel">Clues:</span>
            <ol>
              {committedInputs.map((clue, index) => (
                <li key={`${index}:${clue}`}>
                  <strong>Q{index + 1}</strong> {typeof clue === 'string' ? clue : clue?.text}
                </li>
              ))}
            </ol>
          </section>
        )}

        {currentIntent && (
          <section className="kis-intent-summary" data-testid="kis-intent-summary">
            {currentIntent.query_text && (
              <div className="kis-intent-canonical" data-testid="kis-intent-canonical">
                <span className="kis-intent-label">Resolved Query:</span>
                <span className="kis-intent-query-text">{currentIntent.query_text}</span>
              </div>
            )}

            {Array.isArray(currentIntent.events) && currentIntent.events.length > 0 && (
              <div className="kis-intent-events" data-testid="kis-intent-events">
                <span className="kis-intent-sublabel">Events:</span>
                <div className="kis-events-list">
                  {currentIntent.events.map((event) => (
                    <span key={event.id} className="kis-event-badge" title={event.text}>
                      <strong>{event.id}:</strong> {event.text}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {Array.isArray(currentIntent.entities) && currentIntent.entities.length > 0 && (
              <div className="kis-intent-entities" data-testid="kis-intent-entities">
                <span className="kis-intent-sublabel">Entities:</span>
                <div className="kis-entities-list">
                  {currentIntent.entities.map((entity) => (
                    <span
                      key={entity.id}
                      className="kis-entity-chip"
                      title={`${entity.kind}: ${entity.description}`}
                    >
                      <strong>{entity.id}:</strong> {entity.description}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {Array.isArray(currentIntent.temporal_edges) && currentIntent.temporal_edges.length > 0 && (
              <div className="kis-intent-edges">
                <span className="kis-intent-sublabel">Temporal:</span>
                <div className="kis-edges-list">
                  {currentIntent.temporal_edges.map((edge) => (
                    <span key={`${edge.source}:${edge.target}`} className="kis-edge-badge">
                      {edge.source} ➔ {edge.relation} ➔ {edge.target}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {isSearching && <p className="kis-searching-msg">Searching…</p>}
        {error && <div className="kis-session-error" role="alert">⚠️ {error}</div>}
      </div>

      <div className="kis-chat-footer">
        <div className="kis-chat-input-wrapper">
          <textarea
            ref={inputRef}
            id="event-query"
            className="input-text kis-chat-textarea"
            rows={2}
            value={draft}
            onChange={(event) => onDraftChange?.(event.target.value)}
            placeholder="Search or add another clue…"
            onFocus={onFocusQueryInput}
            onBlur={onBlurQueryInput}
            disabled={isSearching || disabled}
            onKeyDown={handleKeyDown}
          />
        </div>
        <div className="kis-chat-footer-actions">
          {renderExtraActions?.()}
          <button
            type="button"
            className="btn-primary kis-chat-send-btn"
            disabled={isSearching || disabled || !draft.trim()}
            onClick={onSubmit}
          >
            {isSearching ? 'Searching…' : submitLabel}
          </button>
        </div>
      </div>
    </section>
  );
};

export default KisPanel;
