import React, { useRef, useEffect } from 'react';
import IntentSummary from './IntentSummary';
import EventList from './EventList';
import QueryComposer from './QueryComposer';
import FeedbackThread from './FeedbackThread';
import QueryHypothesisEditor from './QueryHypothesisEditor';
import { isResultsStale } from '../queryHypothesisSession';

/**
 * Present one revisioned KIS session as an always-open, ChatGPT-style continuous copilot
 * with Turn 0 search breakdown, interactive multimodal event cards, and conversational feedback turns.
 */
const KisPanel = ({
  sessionState = {},
  feedbackSession = null,
  queryHypothesisState = null,
  onPreviewQueryHypothesis = null,
  onCommitQueryHypothesis = null,
  onCancelPreviewQueryHypothesis = null,
  onUndoQueryHypothesis = null,
  onSearchQueryHypothesis = null,
  onUndoFeedback = null,
  onSelectContext = null,
  onClearContext = null,
  onEditEvent = null,
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
  mode = 'KIS',
}) => {
  const {
    draft = '',
    revision = 0,
    currentIntent = null,
    stagedImages = {},
    isSearching = false,
    error = null,
  } = sessionState;

  const chatBodyRef = useRef(null);

  // Auto-scroll chat body to bottom when new messages arrive or state updates
  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [
    feedbackSession?.messages,
    feedbackSession?.status,
    isSearching,
    currentIntent,
  ]);

  const handleEditEvent = (eventId) => {
    onEditEvent?.(eventId);
    onSelectContext?.(eventId);
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

  // Filter out turn 0 seeds from FeedbackThread since Turn 0 is rendered cleanly in the conversation stream
  const feedbackMessages = (feedbackSession?.messages || []).filter(
    (m) => m.id !== 'msg_initial_query' && m.id !== 'msg_initial_response' && m.id !== 'msg_session_opened'
  );

  const lastPanelAttachTimeRef = useRef(0);

  const handlePanelPaste = (event) => {
    if (event.defaultPrevented || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'INPUT') {
      return;
    }
    const now = Date.now();
    if (now - lastPanelAttachTimeRef.current < 300) {
      return;
    }
    const items = event.clipboardData?.items;
    if (items) {
      for (let i = 0; i < items.length; i += 1) {
        if (items[i].type.startsWith('image/')) {
          const file = items[i].getAsFile();
          if (file) {
            event.preventDefault();
            event.stopPropagation();
            lastPanelAttachTimeRef.current = now;
            const targetId = feedbackSession?.selectedContext?.eventId || 'E1';
            onAttachImage?.(file, targetId);
            return;
          }
        }
      }
    }
    const files = event.clipboardData?.files;
    if (files && files.length > 0) {
      for (let i = 0; i < files.length; i += 1) {
        if (files[i].type.startsWith('image/')) {
          event.preventDefault();
          event.stopPropagation();
          lastPanelAttachTimeRef.current = now;
          const targetId = feedbackSession?.selectedContext?.eventId || 'E1';
          onAttachImage?.(files[i], targetId);
          return;
        }
      }
    }
  };

  return (
    <section
      className="kis-chat-panel"
      data-testid="kis-panel"
      aria-label="KIS search"
      onPaste={handlePanelPaste}
    >
      <div className="kis-chat-header">
        <div className="kis-chat-header-title">
          <span
            className="kis-chat-title-text"
            title={mode === 'AVS' ? 'AVS Ad-Hoc Search' : 'KIS Semantic Search'}
          >
            {mode === 'AVS' ? 'AVS Ad-Hoc Search' : 'KIS Semantic Search'}
          </span>
          {revision > 0 && mode !== 'AVS' && (
            <span className="kis-revision-tag" title={`Revision ${revision}`}>
              Rev {revision}
            </span>
          )}
        </div>
        <div className="kis-chat-header-actions">
          {feedbackSession?.canUndo && onUndoFeedback && (
            <button
              type="button"
              className="kis-chat-undo-btn"
              onClick={onUndoFeedback}
              disabled={feedbackSession?.status === 'pending'}
              title="Undo last feedback turn"
              aria-label="Undo last feedback turn"
            >
              <span className="kis-chat-undo-icon" aria-hidden="true">↺</span>
              <span>Undo</span>
            </button>
          )}
          {onReset && (
            <button
              type="button"
              className="kis-chat-reset-btn"
              onClick={onReset}
              title={mode === 'AVS' ? 'Clear AVS search' : 'Start a new KIS search'}
            >
              {resetLabel}
            </button>
          )}
          {onCollapse && (
            <button
              type="button"
              className="kis-chat-collapse-btn"
              onClick={onCollapse}
              title={mode === 'AVS' ? 'Collapse AVS search panel' : 'Collapse KIS search panel'}
              aria-label={mode === 'AVS' ? 'Collapse AVS search panel' : 'Collapse KIS search panel'}
            >
              ▸
            </button>
          )}
        </div>
      </div>

      <div className="kis-chat-body" ref={chatBodyRef}>
        {/* Empty state when no search has been performed yet */}
        {!currentIntent && (!feedbackSession?.messages || feedbackSession.messages.length === 0) && !isSearching && !error && (
          <div className="kis-chat-empty-state">
            <div className="kis-watermark-badge" data-testid="kis-watermark-badge">
              <img
                src="/hcmus_logo.png"
                alt="HCMUS - Ho Chi Minh University of Science"
                className="kis-watermark-logo"
              />
              <span className="kis-watermark-title">MLeCDanBGold · 2026</span>
            </div>

            <div className="kis-empty-prompt-guide">
              <p className="kis-empty-guide-title">Độ & Ba Gà Claude</p>
              <p className="kis-empty-guide-subtitle">
                {mode === 'AVS'
                  ? 'Search candidate keyframes across the video collection with natural language queries:'
                  : 'Describe visual events across time, attach reference images, or pick a starter clue:'}
              </p>
              <div className="kis-quick-prompts">
                {mode === 'AVS' ? (
                  <>
                    <button
                      type="button"
                      className="kis-quick-prompt-btn"
                      onClick={() => onDraftChange?.('A red sports car driving on a road')}
                    >
                      <span className="prompt-text">A red sports car driving on a road</span>
                    </button>
                    <button
                      type="button"
                      className="kis-quick-prompt-btn"
                      onClick={() => onDraftChange?.('People celebrating on a stage with lights')}
                    >
                      <span className="prompt-text">People celebrating on a stage</span>
                    </button>
                    <button
                      type="button"
                      className="kis-quick-prompt-btn"
                      onClick={() => onDraftChange?.('A dog jumping into a swimming pool')}
                    >
                      <span className="prompt-text">Dog jumping into a pool</span>
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      type="button"
                      className="kis-quick-prompt-btn"
                      onClick={() => onDraftChange?.('Find a person wearing yellow jacket walking a dog')}
                    >
                      <span className="prompt-text">Person in yellow jacket walking dog</span>
                    </button>
                    <button
                      type="button"
                      className="kis-quick-prompt-btn"
                      onClick={() => onDraftChange?.('Two cars collide at a crossroads')}
                    >
                      <span className="prompt-text">Two cars collide at a crossroads</span>
                    </button>
                    <button
                      type="button"
                      className="kis-quick-prompt-btn"
                      onClick={() => onDraftChange?.('E1: Chef cuts vegetables\nE2: Chef cooks on pan')}
                    >
                      <span className="prompt-text">Multi-event cooking sequence</span>
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Fallback conversation stream when there is no currentIntent (e.g. AVS mode) */}
        {!currentIntent && (feedbackSession?.messages || []).length > 0 && (
          <div className="kis-conversation-stream">
            <FeedbackThread
              messages={feedbackSession.messages}
              canUndo={false}
              isPending={false}
            />
          </div>
        )}

        {/* Unified Conversation Stream */}
        {currentIntent && (
          <div className="kis-conversation-stream">
            {/* Hidden metadata container preserving data-testids for IntentSummary */}
            <div className="kis-intent-summary-wrapper" style={{ display: 'none' }}>
              <IntentSummary currentIntent={currentIntent} />
            </div>

            {/* Turn 0: User Query Bubble */}
            <div className="feedback-message feedback-message-user kis-turn0-user">
              <div className="feedback-message-meta">
                <span className="feedback-role-label">You</span>
              </div>
              <div className="feedback-message-body" data-testid="kis-intent-canonical">
                {currentIntent.query_text || 'Multimodal search'}
              </div>
            </div>

            {/* Turn 0: Assistant Breakdown with Interactive Event Cards */}
            <div className="feedback-message feedback-message-assistant kis-turn0-assistant">
              <div className="feedback-message-meta">
                <span className="feedback-role-label">Assistant</span>
                {typeof currentIntent.revision === 'number' && currentIntent.revision > 0 && (
                  <span className="feedback-chip chip-scope">Rev {currentIntent.revision}</span>
                )}
                {Array.isArray(currentIntent.events) && (
                  <span className="feedback-chip chip-events">
                    {currentIntent.events.length} Event{currentIntent.events.length === 1 ? '' : 's'}
                  </span>
                )}
              </div>

              <div className="feedback-message-body">
                {queryHypothesisState?.intent ? (
                  <QueryHypothesisEditor
                    intent={queryHypothesisState.intent}
                    preview={queryHypothesisState.preview}
                    canUndo={Boolean(queryHypothesisState.canUndo)}
                    onPreviewAction={onPreviewQueryHypothesis}
                    onCommit={onCommitQueryHypothesis}
                    onCancelPreview={onCancelPreviewQueryHypothesis}
                    onUndo={onUndoQueryHypothesis}
                    onSearch={onSearchQueryHypothesis || onSubmit}
                    isResultsStale={isResultsStale(queryHypothesisState)}
                    disabled={isSearching || disabled}
                    onAttachImage={onAttachImage}
                    onRemoveImage={onRemoveImage}
                  />
                ) : (
                  <EventList
                    events={currentIntent.events || []}
                    stagedImages={stagedImages}
                    onEdit={handleEditEvent}
                    onAddImage={handleAddImageToEvent}
                    onRemoveImage={onRemoveImage}
                    onSelectContext={onSelectContext}
                    selectedEventId={feedbackSession?.selectedContext?.eventId}
                    disabled={isSearching || disabled}
                  />
                )}

                {((Array.isArray(currentIntent.entities) && currentIntent.entities.length > 0)
                  || (Array.isArray(currentIntent.temporal_edges) && currentIntent.temporal_edges.length > 0)) && (
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
              </div>
            </div>

            {/* Subsequent Conversational Feedback Turns */}
            {feedbackSession && (
              <FeedbackThread
                messages={feedbackMessages}
                canUndo={!!feedbackSession.canUndo}
                onUndo={onUndoFeedback}
                isPending={feedbackSession.status === 'pending'}
              />
            )}
          </div>
        )}

        {isSearching && !currentIntent && <p className="kis-searching-msg">Searching…</p>}
        {error && <div className="kis-session-error" role="alert">⚠️ {error}</div>}
      </div>

      <div className="kis-chat-footer">
        <QueryComposer
          draft={draft}
          onDraftChange={onDraftChange}
          onSubmit={onSubmit}
          onAttachImage={onAttachImage}
          onRemoveImage={onRemoveImage}
          baseIntent={currentIntent}
          stagedImages={stagedImages}
          isSearching={isSearching}
          disabled={disabled}
          inputRef={inputRef}
          onFocus={onFocusQueryInput}
          onBlur={onBlurQueryInput}
          renderExtraActions={renderExtraActions}
          submitLabel={submitLabel}
          placeholder={mode === 'AVS' ? 'Enter an ad-hoc search query…' : undefined}
          selectedContext={feedbackSession?.selectedContext}
          onClearContext={onClearContext}
        />
      </div>
    </section>
  );
};

export default KisPanel;
