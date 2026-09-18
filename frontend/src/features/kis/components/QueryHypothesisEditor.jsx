import React, { useState } from 'react';
import QueryHypothesisPreview from './QueryHypothesisPreview';

/**
 * Structured editor for KIS Query Hypotheses.
 * Supports Split, Merge, Reorder, Edit, Add, and Undo operations
 * via preview and commit stages without automatically running search.
 */
const QueryHypothesisEditor = ({
  intent,
  preview = null,
  pendingAction = null,
  canUndo = false,
  onPreviewAction,
  onCommit,
  onApplyPreview,
  onCancelPreview,
  onUndo,
  onSearch,
  isResultsStale = false,
  disabled = false,
}) => {
  const [editingEventId, setEditingEventId] = useState(null);
  const [editText, setEditText] = useState('');

  const [splittingEventId, setSplittingEventId] = useState(null);
  const [splitIndex, setSplitIndex] = useState('');

  const [addingAtPosition, setAddingAtPosition] = useState(null);
  const [addText, setAddText] = useState('');

  if (!intent) return null;

  const events = intent.events || [];

  const handleStartEdit = (event) => {
    setEditingEventId(event.id);
    setEditText(event.text || '');
    setSplittingEventId(null);
    setAddingAtPosition(null);
  };

  const handleSaveEdit = (eventId) => {
    if (!editText.trim()) return;
    const action = {
      type: 'edit_event',
      event_id: eventId,
      text: editText.trim(),
    };
    onPreviewAction?.(action);
    setEditingEventId(null);
  };

  const handleStartSplit = (event) => {
    setSplittingEventId(event.id);
    const mid = Math.floor((event.text?.length || 0) / 2);
    setSplitIndex(String(mid > 0 ? mid : 1));
    setEditingEventId(null);
    setAddingAtPosition(null);
  };

  const handleConfirmSplit = (eventId) => {
    const idx = parseInt(splitIndex, 10);
    if (isNaN(idx) || idx <= 0) return;
    const action = {
      type: 'split_event',
      event_id: eventId,
      split_at: idx,
      image_assignments: {},
    };
    onPreviewAction?.(action);
    setSplittingEventId(null);
  };

  const handleMerge = (leftEventId, rightEventId) => {
    const action = {
      type: 'merge_events',
      left_event_id: leftEventId,
      right_event_id: rightEventId,
    };
    onPreviewAction?.(action);
  };

  const handleMove = (currentIndex, direction) => {
    const targetIndex = currentIndex + direction;
    if (targetIndex < 0 || targetIndex >= events.length) return;
    const reordered = [...events];
    const [moved] = reordered.splice(currentIndex, 1);
    reordered.splice(targetIndex, 0, moved);
    const action = {
      type: 'reorder_events',
      event_ids: reordered.map((e) => e.id),
    };
    onPreviewAction?.(action);
  };

  const handleAdd = (position) => {
    if (!addText.trim()) return;
    const action = {
      type: 'add_event',
      position,
      text: addText.trim(),
      images: [],
    };
    onPreviewAction?.(action);
    setAddingAtPosition(null);
    setAddText('');
  };

  const handleApply = () => {
    if (onCommit) {
      onCommit(pendingAction || preview);
    } else if (onApplyPreview) {
      onApplyPreview();
    }
  };

  return (
    <section
      className="query-hypothesis-editor"
      data-testid="query-hypothesis-editor"
      aria-label="Query Hypothesis Editor"
    >
      <div className="query-hypothesis-header">
        <div className="query-hypothesis-title-row">
          <h3>Query Hypothesis</h3>
          <span className="query-revision-badge">Rev {intent.revision}</span>
          {intent.language && (
            <span className="query-language-badge">{intent.language.toUpperCase()}</span>
          )}
        </div>

        {intent.query_text && (
          <p className="query-canonical-text" title="Original canonical query">
            &ldquo;{intent.query_text}&rdquo;
          </p>
        )}

        <div className="query-hypothesis-toolbar">
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary query-undo-btn"
            onClick={onUndo}
            disabled={disabled || !canUndo}
            title="Revert last query hypothesis modification"
          >
            ↺ Undo
          </button>

          <button
            type="button"
            className="btn btn-sm btn-outline-primary query-add-start-btn"
            onClick={() => setAddingAtPosition(0)}
            disabled={disabled}
          >
            + Add Event at Start
          </button>
        </div>
      </div>

      {isResultsStale && (
        <div
          className="query-hypothesis-stale-notice alert alert-warning"
          data-testid="stale-results-notice"
        >
          <span>Query changed since last search. Results may be out of date.</span>
          {onSearch && (
            <button
              type="button"
              className="btn btn-sm btn-warning query-stale-search-btn"
              onClick={onSearch}
              disabled={disabled}
            >
              Search
            </button>
          )}
        </div>
      )}

      {preview && (
        <QueryHypothesisPreview
          preview={preview}
          currentIntent={intent}
          onApply={handleApply}
          onCancel={onCancelPreview}
          disabled={disabled}
        />
      )}

      {addingAtPosition !== null && (
        <div className="query-event-inline-form adding-form">
          <input
            type="text"
            className="form-control"
            placeholder="New event description..."
            value={addText}
            onChange={(e) => setAddText(e.target.value)}
            disabled={disabled}
            autoFocus
          />
          <div className="form-buttons">
            <button
              type="button"
              className="btn btn-sm btn-primary"
              onClick={() => handleAdd(addingAtPosition)}
              disabled={disabled || !addText.trim()}
            >
              Add
            </button>
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={() => {
                setAddingAtPosition(null);
                setAddText('');
              }}
              disabled={disabled}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="query-hypothesis-events">
        {events.map((event, index) => {
          const isEditing = editingEventId === event.id;
          const isSplitting = splittingEventId === event.id;
          const originLabel =
            event.origin === 'source'
              ? 'Source'
              : event.origin === 'user_override'
              ? 'Edited'
              : event.origin === 'user_added'
              ? 'Added'
              : event.origin || 'Source';

          return (
            <div
              key={event.id}
              className={`query-hypothesis-event-item ${event.origin ? `origin-${event.origin}` : ''}`}
              data-testid={`hypothesis-event-${event.id}`}
            >
              <div className="event-item-main">
                <div className="event-item-header">
                  <span className="event-id-tag">{event.id}</span>
                  <span className={`event-origin-pill pill-${event.origin || 'source'}`}>
                    {originLabel}
                  </span>
                  {event.source_provenance && (
                    <span
                      className="event-provenance-info"
                      title={`Characters ${event.source_provenance.start_char}..${event.source_provenance.end_char} in query`}
                    >
                      [{event.source_provenance.start_char}:{event.source_provenance.end_char}]
                    </span>
                  )}
                </div>

                {isEditing ? (
                  <div className="query-event-inline-form edit-form">
                    <input
                      type="text"
                      className="form-control"
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      disabled={disabled}
                      autoFocus
                    />
                    <div className="form-buttons">
                      <button
                        type="button"
                        className="btn btn-sm btn-primary"
                        onClick={() => handleSaveEdit(event.id)}
                        disabled={disabled || !editText.trim()}
                      >
                        Preview Edit
                      </button>
                      <button
                        type="button"
                        className="btn btn-sm btn-secondary"
                        onClick={() => setEditingEventId(null)}
                        disabled={disabled}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : isSplitting ? (
                  <div className="query-event-inline-form split-form">
                    <div className="split-text-preview">
                      <span className="left-half">
                        {event.text.slice(0, parseInt(splitIndex, 10) || 0)}
                      </span>
                      <span className="split-divider"> | </span>
                      <span className="right-half">
                        {event.text.slice(parseInt(splitIndex, 10) || 0)}
                      </span>
                    </div>
                    <label className="split-input-label">
                      Split character index:
                      <input
                        type="number"
                        min="1"
                        max={Math.max(1, (event.text?.length || 1) - 1)}
                        className="form-control split-index-input"
                        value={splitIndex}
                        onChange={(e) => setSplitIndex(e.target.value)}
                        disabled={disabled}
                      />
                    </label>
                    <div className="form-buttons">
                      <button
                        type="button"
                        className="btn btn-sm btn-primary"
                        onClick={() => handleConfirmSplit(event.id)}
                        disabled={
                          disabled ||
                          !splitIndex ||
                          parseInt(splitIndex, 10) <= 0 ||
                          parseInt(splitIndex, 10) >= event.text.length
                        }
                      >
                        Preview Split
                      </button>
                      <button
                        type="button"
                        className="btn btn-sm btn-secondary"
                        onClick={() => setSplittingEventId(null)}
                        disabled={disabled}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <p className="event-item-text">{event.text}</p>
                )}
              </div>

              {!isEditing && !isSplitting && (
                <div className="event-item-controls">
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-secondary"
                    onClick={() => handleStartEdit(event)}
                    disabled={disabled}
                    title="Edit event text"
                  >
                    Edit
                  </button>

                  <button
                    type="button"
                    className="btn btn-sm btn-outline-secondary"
                    onClick={() => handleStartSplit(event)}
                    disabled={disabled || (event.text?.length || 0) <= 1}
                    title="Split this event into two sequential events"
                  >
                    Split
                  </button>

                  {index < events.length - 1 && (
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary"
                      onClick={() => handleMerge(event.id, events[index + 1].id)}
                      disabled={disabled}
                      title={`Merge with ${events[index + 1].id}`}
                    >
                      Merge with Next
                    </button>
                  )}

                  <div className="reorder-buttons">
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary"
                      onClick={() => handleMove(index, -1)}
                      disabled={disabled || index === 0}
                      title="Move event earlier"
                    >
                      ▲
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary"
                      onClick={() => handleMove(index, 1)}
                      disabled={disabled || index === events.length - 1}
                      title="Move event later"
                    >
                      ▼
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
};

export default QueryHypothesisEditor;
