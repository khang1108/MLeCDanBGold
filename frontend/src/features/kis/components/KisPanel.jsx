import React, { useEffect, useRef } from 'react';

/**
 * Unified KIS Chat Panel displaying conversation message bubbles,
 * committed clue history, semantic intent breakdown card (canonical query,
 * events, entities, temporal relations), and chat input.
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

  const chatBottomRef = useRef(null);

  useEffect(() => {
    if (typeof chatBottomRef.current?.scrollIntoView === 'function') {
      chatBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [committedInputs.length, currentIntent, isSearching]);

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!isSearching && draft.trim()) {
        onSubmit?.(event);
      }
    }
  };

  return (
    <div className="kis-chat-panel" data-testid="kis-chat-panel">
      {/* Header */}
      <div className="kis-chat-header">
        <div className="kis-chat-header-title">
          <span className="kis-chat-sparkle" aria-hidden="true">✨</span>
          <span className="kis-chat-title-text">KIS Assistant</span>
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
            title="Reset clue session and start a new search"
          >
            {resetLabel}
          </button>
        )}
      </div>

      {/* Scrollable chat body */}
      <div className="kis-chat-body">
        {/* Welcome bubble when starting */}
        {committedInputs.length === 0 && (
          <div className="kis-chat-message is-assistant">
            <div className="kis-chat-avatar" aria-hidden="true">🤖</div>
            <div className="kis-chat-bubble assistant-bubble">
              <p className="kis-welcome-title"><strong>Multi-Clue Semantic Search</strong></p>
              <p className="kis-welcome-desc">
                Enter your query or clue to begin. Add more clues to decompose your search into sequential events and entities.
              </p>
            </div>
          </div>
        )}

        {/* Committed clues rendered as message pairs */}
        {committedInputs.map((clue, index) => {
          const text = typeof clue === 'string' ? clue : clue?.text;
          const isLast = index === committedInputs.length - 1;
          const clueNumber = index + 1;

          return (
            <React.Fragment key={index}>
              {/* User message bubble (right-aligned) */}
              <div className="kis-chat-message is-user">
                <div className="kis-chat-bubble user-bubble">
                  <span className="kis-clue-badge">Q{clueNumber}</span>
                  <span className="kis-clue-text">{text}</span>
                </div>
              </div>

              {/* Assistant response bubble (left-aligned) */}
              <div className="kis-chat-message is-assistant">
                <div className="kis-chat-avatar" aria-hidden="true">🤖</div>
                <div className="kis-chat-bubble assistant-bubble">
                  <p className="kis-assistant-status">
                    {isLast && currentIntent
                      ? `Resolved intent for clue Q${clueNumber}:`
                      : `Clue Q${clueNumber} processed.`}
                  </p>
                </div>
              </div>

              {/* Big Intent Breakdown Card for the latest clue */}
              {isLast && currentIntent && (
                <div className="kis-chat-intent-wrapper">
                  <div className="kis-intent-summary" data-testid="kis-intent-summary">
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
                          {currentIntent.events.map((ev) => (
                            <span key={ev.id} className="kis-event-badge" title={ev.text}>
                              <strong>{ev.id}:</strong> {ev.text}
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
                          {currentIntent.temporal_edges.map((edge, edgeIdx) => (
                            <span key={edgeIdx} className="kis-edge-badge">
                              {edge.source} ➔ {edge.relation} ➔ {edge.target}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </React.Fragment>
          );
        })}

        {/* Searching indicator */}
        {isSearching && (
          <div className="kis-chat-message is-assistant">
            <div className="kis-chat-avatar" aria-hidden="true">🤖</div>
            <div className="kis-chat-bubble assistant-bubble is-searching">
              <span className="kis-dot" />
              <span className="kis-dot" />
              <span className="kis-dot" />
              <span className="kis-searching-msg">Decomposing clues and searching…</span>
            </div>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="kis-session-error" role="alert">
            ⚠️ {error}
          </div>
        )}

        <div ref={chatBottomRef} />
      </div>

      {/* Chat input footer */}
      <div className="kis-chat-footer">
        <div className="kis-chat-input-wrapper">
          <textarea
            ref={inputRef}
            id="kis-chat-query-input"
            className="input-text kis-chat-textarea"
            rows={2}
            value={draft}
            onChange={(e) => onDraftChange?.(e.target.value)}
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
    </div>
  );
};

export default KisPanel;
