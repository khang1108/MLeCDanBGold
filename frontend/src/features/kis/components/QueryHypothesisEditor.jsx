import React, { useState } from 'react';
import QueryHypothesisPreview from './QueryHypothesisPreview';
import { kisImageAssetUrl } from '../../../api/kis';

/**
 * Compute valid interactive split boundaries in text (word boundaries or character boundaries).
 */
export const computeTextBoundaries = (text) => {
  if (!text || text.length <= 1) return { type: 'none', words: [], chars: [], boundaries: [] };

  const wordsWithOffsets = [];
  const regex = /\S+/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    wordsWithOffsets.push({
      word: match[0],
      start: match.index,
      end: match.index + match[0].length,
    });
  }

  if (wordsWithOffsets.length > 1) {
    const boundaries = [];
    for (let i = 0; i < wordsWithOffsets.length - 1; i += 1) {
      const currentWord = wordsWithOffsets[i];
      const nextWord = wordsWithOffsets[i + 1];
      boundaries.push({
        index: nextWord.start,
        prevWord: currentWord.word,
        nextWord: nextWord.word,
        label: `${currentWord.word} | ${nextWord.word}`,
      });
    }
    return {
      type: 'words',
      words: wordsWithOffsets,
      chars: [],
      boundaries,
    };
  }

  // Single word: character boundaries
  const chars = text.split('');
  const boundaries = [];
  for (let i = 1; i < chars.length; i += 1) {
    boundaries.push({
      index: i,
      prevWord: chars.slice(0, i).join(''),
      nextWord: chars.slice(i).join(''),
      label: `${chars[i - 1]} | ${chars[i]}`,
    });
  }
  return {
    type: 'chars',
    words: [],
    chars,
    boundaries,
  };
};

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
  onAttachImage,
  onRemoveImage,
}) => {
  const [editingEventId, setEditingEventId] = useState(null);
  const [editText, setEditText] = useState('');

  const [splittingEventId, setSplittingEventId] = useState(null);
  const [splitIndex, setSplitIndex] = useState('');
  const [splitImageAssignments, setSplitImageAssignments] = useState({});

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
      type: 'edit',
      event_id: eventId,
      text: editText.trim(),
    };
    onPreviewAction?.(action);
    setEditingEventId(null);
  };

  const handleStartSplit = (event) => {
    setSplittingEventId(event.id);
    const text = event.text || '';
    const boundaryInfo = computeTextBoundaries(text);
    let initialIndex = 1;
    if (boundaryInfo.boundaries.length > 0) {
      const midBoundary = boundaryInfo.boundaries[Math.floor(boundaryInfo.boundaries.length / 2)];
      initialIndex = midBoundary.index;
    } else {
      initialIndex = Math.max(1, Math.floor(text.length / 2));
    }
    setSplitIndex(String(initialIndex));
    setSplitImageAssignments({});
    setEditingEventId(null);
    setAddingAtPosition(null);
  };

  const handleConfirmSplit = (eventId) => {
    const idx = parseInt(splitIndex, 10);
    if (isNaN(idx) || idx <= 0) return;
    const event = events.find((candidate) => candidate.id === eventId);
    const imageAssignments = {};
    for (const image of event?.images || []) {
      const assetId = typeof image === 'string' ? image : (image.asset_id || image.id);
      if (assetId && splitImageAssignments[assetId]?.length) {
        imageAssignments[assetId] = splitImageAssignments[assetId];
      }
    }
    const action = {
      type: 'split',
      event_id: eventId,
      split_at: idx,
      image_assignments: imageAssignments,
    };
    onPreviewAction?.(action);
    setSplittingEventId(null);
    setSplitImageAssignments({});
  };

  const handleMerge = (leftEventId, rightEventId) => {
    const action = {
      type: 'merge',
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
      type: 'reorder',
      event_ids: reordered.map((e) => e.id),
    };
    onPreviewAction?.(action);
  };

  const handleAdd = (position) => {
    if (!addText.trim()) return;
    const action = {
      type: 'add',
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
          <div className="title-with-badges">
            <h3>Query Hypothesis</h3>
            <span className="query-revision-badge">Rev {intent.revision}</span>
            {intent.language && (
              <span className="query-language-badge">{intent.language.toUpperCase()}</span>
            )}
          </div>
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
            <span className="btn-icon">↺</span>
            <span>Undo</span>
          </button>

          <button
            type="button"
            className="btn btn-sm btn-outline-primary query-add-start-btn"
            onClick={() => setAddingAtPosition(0)}
            disabled={disabled}
          >
            <span className="btn-icon">+</span>
            <span>Add Event at Start</span>
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
                    <div className="split-text-preview" aria-live="polite">
                      <div className="split-preview-header">
                        <span className="split-preview-title">Split Preview</span>
                      </div>
                      <div className="split-preview-halves">
                        <div className="split-preview-child left-child">
                          <span className="split-child-badge">Event 1</span>
                          <span className="split-child-text">
                            {event.text.slice(0, parseInt(splitIndex, 10) || 0).trim() || '...'}
                          </span>
                        </div>
                        <div className="split-divider-symbol" aria-hidden="true">
                          ✂
                        </div>
                        <div className="split-preview-child right-child">
                          <span className="split-child-badge">Event 2</span>
                          <span className="split-child-text">
                            {event.text.slice(parseInt(splitIndex, 10) || 0).trim() || '...'}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="split-boundary-selector">
                      <div className="split-selector-header">
                        <span className="split-boundary-instruction">
                          Click any boundary or word to set split point:
                        </span>
                      </div>
                      <div className="split-tokens-flow">
                        {(() => {
                          const boundaryInfo = computeTextBoundaries(event.text);
                          const currentSplit = parseInt(splitIndex, 10) || 0;

                          if (boundaryInfo.type === 'words') {
                            return boundaryInfo.words.map((w, wIdx) => {
                              const boundary = wIdx > 0 ? boundaryInfo.boundaries[wIdx - 1] : null;
                              const isBoundarySelected = boundary && boundary.index === currentSplit;
                              const isLeft = w.end <= currentSplit;

                              return (
                                <React.Fragment key={`word-${w.start}-${w.end}`}>
                                  {boundary && (
                                    <button
                                      type="button"
                                      className={`split-boundary-btn ${isBoundarySelected ? 'is-selected' : ''}`}
                                      onClick={() => setSplitIndex(String(boundary.index))}
                                      aria-label={`Split between "${boundary.prevWord}" and "${boundary.nextWord}"`}
                                      title={`Split between "${boundary.prevWord}" and "${boundary.nextWord}"`}
                                      disabled={disabled}
                                    >
                                      <span className="split-cut-icon">{isBoundarySelected ? '✂' : '|'}</span>
                                    </button>
                                  )}
                                  <button
                                    type="button"
                                    className={`split-word-token ${isLeft ? 'token-left' : 'token-right'}`}
                                    onClick={() => {
                                      const targetBoundary = boundaryInfo.boundaries.find((b) => b.index >= w.end)
                                        || boundaryInfo.boundaries[boundaryInfo.boundaries.length - 1];
                                      if (targetBoundary) {
                                        setSplitIndex(String(targetBoundary.index));
                                      }
                                    }}
                                    title={`Click to split after "${w.word}"`}
                                    disabled={disabled}
                                  >
                                    {w.word}
                                  </button>
                                </React.Fragment>
                              );
                            });
                          }

                          if (boundaryInfo.type === 'chars') {
                            return boundaryInfo.chars.map((char, cIdx) => {
                              const boundary = cIdx > 0 ? boundaryInfo.boundaries[cIdx - 1] : null;
                              const isBoundarySelected = boundary && boundary.index === currentSplit;
                              const isLeft = cIdx < currentSplit;

                              return (
                                <React.Fragment key={`char-${cIdx}`}>
                                  {boundary && (
                                    <button
                                      type="button"
                                      className={`split-boundary-btn char-boundary ${isBoundarySelected ? 'is-selected' : ''}`}
                                      onClick={() => setSplitIndex(String(boundary.index))}
                                      aria-label={`Split between '${boundary.prevWord.slice(-1)}' and '${boundary.nextWord[0]}'`}
                                      title="Split here"
                                      disabled={disabled}
                                    >
                                      <span className="split-cut-icon">{isBoundarySelected ? '✂' : '|'}</span>
                                    </button>
                                  )}
                                  <span className={`split-char-token ${isLeft ? 'token-left' : 'token-right'}`}>
                                    {char}
                                  </span>
                                </React.Fragment>
                              );
                            });
                          }

                          return null;
                        })()}
                      </div>
                    </div>
                    {(event.images || []).length > 0 && (
                      <fieldset className="split-image-assignments">
                        <legend>Assign each image to split children</legend>
                        {(event.images || []).map((image) => {
                          const assetId = typeof image === 'string' ? image : (image.asset_id || image.id);
                          const assigned = splitImageAssignments[assetId] || [];
                          return (
                            <div key={assetId} className="split-image-assignment-row">
                              <span>{assetId}</span>
                              {['left', 'right'].map((side) => (
                                <label key={side}>
                                  <input
                                    type="checkbox"
                                    aria-label={`Assign ${assetId} to ${side} child`}
                                    checked={assigned.includes(side)}
                                    onChange={() => setSplitImageAssignments((previous) => {
                                      const current = previous[assetId] || [];
                                      const next = current.includes(side)
                                        ? current.filter((value) => value !== side)
                                        : [...current, side];
                                      return { ...previous, [assetId]: next };
                                    })}
                                    disabled={disabled}
                                  />
                                  {side}
                                </label>
                              ))}
                            </div>
                          );
                        })}
                      </fieldset>
                    )}
                    <div className="form-buttons">
                      <button
                        type="button"
                        className="btn btn-sm btn-primary"
                        onClick={() => handleConfirmSplit(event.id)}
                        disabled={
                          disabled ||
                          !splitIndex ||
                          parseInt(splitIndex, 10) <= 0 ||
                          parseInt(splitIndex, 10) >= event.text.length ||
                          (event.images || []).some((image) => {
                            const assetId = typeof image === 'string' ? image : (image.asset_id || image.id);
                            return !assetId || !(splitImageAssignments[assetId] || []).length;
                          })
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
                  <>
                    <p className="event-item-text">{event.text}</p>
                    {Array.isArray(event.images) && event.images.length > 0 && (
                      <div className="query-event-images" data-testid={`event-images-${event.id}`}>
                        {event.images.map((image, imgIdx) => {
                          const assetId = typeof image === 'string' ? image : (image.asset_id || image.id);
                          const fileName = typeof image === 'object' ? (image.filename || image.file_name || assetId) : assetId;
                          return (
                            <div key={assetId || imgIdx} className="query-event-image-chip" data-testid={`event-image-${assetId}`}>
                              {assetId && (
                                <img
                                  src={kisImageAssetUrl(assetId)}
                                  alt={fileName}
                                  className="query-event-image-thumb"
                                />
                              )}
                              <span className="query-event-image-name">{fileName}</span>
                              <button
                                type="button"
                                className="query-event-remove-image-btn"
                                aria-label={`Remove image ${fileName}`}
                                title={`Remove image ${fileName}`}
                                onClick={() => onRemoveImage?.(event.id, assetId)}
                                disabled={disabled}
                              >
                                ×
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </>
                )}
              </div>

              {!isEditing && !isSplitting && (
                <div className="event-item-controls">
                  <div className="event-action-group">
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary event-ctrl-btn btn-edit"
                      onClick={() => handleStartEdit(event)}
                      disabled={disabled}
                      title="Edit event text"
                    >
                      <span className="btn-icon">✎</span>
                      <span>Edit</span>
                    </button>

                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary event-ctrl-btn btn-attach-img"
                      onClick={() => {
                        const input = document.createElement('input');
                        input.type = 'file';
                        input.accept = 'image/*';
                        input.onchange = (e) => {
                          const file = e.target.files?.[0];
                          if (file) {
                            onAttachImage?.(file, event.id);
                          }
                        };
                        input.click();
                      }}
                      disabled={disabled}
                      title="Attach image to this event"
                      aria-label={`Attach image to ${event.id}`}
                    >
                      <span className="btn-icon">📷</span>
                      <span>Image</span>
                    </button>

                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary event-ctrl-btn btn-split"
                      onClick={() => handleStartSplit(event)}
                      disabled={disabled || (event.text?.length || 0) <= 1}
                      title="Split this event into two sequential events"
                    >
                      <span className="btn-icon">✂</span>
                      <span>Split</span>
                    </button>

                    {index < events.length - 1 && (
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-secondary event-ctrl-btn btn-merge"
                        onClick={() => handleMerge(event.id, events[index + 1].id)}
                        disabled={disabled}
                        title={`Merge with ${events[index + 1].id}`}
                      >
                        <span className="btn-icon">⧉</span>
                        <span>Merge with Next</span>
                      </button>
                    )}
                  </div>

                  <div className="reorder-buttons">
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary reorder-btn"
                      onClick={() => handleMove(index, -1)}
                      disabled={disabled || index === 0}
                      title="Move event earlier"
                      aria-label="Move event earlier"
                    >
                      ▲
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary reorder-btn"
                      onClick={() => handleMove(index, 1)}
                      disabled={disabled || index === events.length - 1}
                      title="Move event later"
                      aria-label="Move event later"
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
