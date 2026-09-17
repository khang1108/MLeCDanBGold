import React from 'react';
import IntentSummary from './IntentSummary';
import EventList from './EventList';
import QueryComposer from './QueryComposer';
import FeedbackThread from './FeedbackThread';

/**
 * Present one revisioned KIS session as multimodal event cards, canonical intent,
 * and unified query composer.
 */
const KisPanel = ({
  sessionState = {},
  feedbackSession = null,
  onUndoFeedback = null,
  onSelectContext = null,
  onClearContext = null,
  onDraftChange,
  onSubmit,
  onReset,
  onAttachImage,
  onRemoveImage,
  disabled = false,
  inputRef,
  onFocusQueryInput,
  onBlurQueryInput,
  renderExtraActions,
  submitLabel = 'Search',
  resetLabel = 'New Search',
  onCollapse,
}) => {
  const {
    draft = '',
    revision = 0,
    currentIntent = null,
    stagedImages = {},
    isSearching = false,
    error = null,
  } = sessionState;

  const handleEditEvent = (eventId) => {
    const prefix = `${eventId}: `;
    onDraftChange?.(prefix);
    setTimeout(() => {
      inputRef?.current?.focus();
    }, 0);
  };

  const handleAddImageToEvent = (eventId) => {
    // Open file picker specifically for this event
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.onchange = (e) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        onAttachImage?.(files[0], eventId);
      }
    };
    fileInput.click();
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
        <div className="kis-chat-header-actions">
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
          {onCollapse && (
            <button
              type="button"
              className="kis-chat-collapse-btn"
              onClick={onCollapse}
              title="Collapse KIS search panel"
              aria-label="Collapse KIS search panel"
            >
              ▸
            </button>
          )}
        </div>
      </div>

      <div className="kis-chat-body">
        <IntentSummary currentIntent={currentIntent} />

        <EventList
          events={currentIntent?.events || []}
          stagedImages={stagedImages}
          onEdit={handleEditEvent}
          onAddImage={handleAddImageToEvent}
          onRemoveImage={onRemoveImage}
          onSelectContext={onSelectContext}
          selectedEventId={feedbackSession?.selectedContext?.eventId}
          disabled={isSearching || disabled}
        />

        {feedbackSession && (
          <FeedbackThread
            messages={feedbackSession.messages || []}
            canUndo={!!feedbackSession.canUndo}
            onUndo={onUndoFeedback}
            isPending={feedbackSession.status === 'pending'}
          />
        )}

        {currentIntent && (
          (Array.isArray(currentIntent.entities) && currentIntent.entities.length > 0)
          || (Array.isArray(currentIntent.temporal_edges) && currentIntent.temporal_edges.length > 0)
        ) && (
          <details className="kis-semantic-details" data-testid="kis-semantic-details">
            <summary className="kis-semantic-summary">Semantic details</summary>
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
          </details>
        )}

        {!currentIntent && !isSearching && !error && (
          <div className="kis-watermark-badge" data-testid="kis-watermark-badge">
            <img
              src="/hcmus_logo.png"
              alt="HCMUS - Ho Chi Minh University of Science"
              className="kis-watermark-logo"
            />
            <span className="kis-watermark-title">MLeCDanBGold · 2026</span>
          </div>
        )}

        {isSearching && <p className="kis-searching-msg">Searching…</p>}
        {error && <div className="kis-session-error" role="alert">⚠️ {error}</div>}
      </div>

      <div className="kis-chat-footer">
        <QueryComposer
          draft={draft}
          onDraftChange={onDraftChange}
          onSubmit={onSubmit}
          onAttachImage={onAttachImage}
          baseIntent={currentIntent}
          stagedImages={stagedImages}
          isSearching={isSearching}
          disabled={disabled}
          inputRef={inputRef}
          onFocus={onFocusQueryInput}
          onBlur={onBlurQueryInput}
          renderExtraActions={renderExtraActions}
          submitLabel={submitLabel}
          selectedContext={feedbackSession?.selectedContext}
          onClearContext={onClearContext}
        />
      </div>
    </section>
  );
};

export default KisPanel;
