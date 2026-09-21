import React, { useState, useRef } from 'react';
import { parseComposerDraft } from '../parser';
import { kisImageAssetUrl } from '../../../api/kis';

/**
 * Unified multimodal composer: textual instructions, staged images, preview, and dynamic action verbs.
 */
const QueryComposer = ({
  draft = '',
  onDraftChange,
  onSubmit,
  onAttachImage,
  onRemoveImage,
  baseIntent = null,
  stagedImages = {},
  isSearching = false,
  disabled = false,
  inputRef,
  onFocus,
  onBlur,
  renderExtraActions,
  submitLabel = 'Search',
  placeholder = 'Search or add another clue…',
  selectedContext = null,
  onClearContext = null,
}) => {
  const [pendingImageFile, setPendingImageFile] = useState(null);
  const localTextareaRef = useRef(null);
  const textareaRef = inputRef || localTextareaRef;

  const events = Array.isArray(baseIntent?.events) ? baseIntent.events : [];
  const hasStagedImages = Object.keys(stagedImages || {}).length > 0;
  const preview = draft.trim() ? parseComposerDraft(draft, baseIntent) : null;

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!isSearching && !disabled) {
        onSubmit?.(event);
      }
    }
  };

  const resolveTargetEventId = (file) => {
    // If user clicked/selected a specific event context in the thread:
    if (selectedContext?.eventId) {
      return selectedContext.eventId;
    }
    // If draft has an explicit scoped header E#:
    const headerMatch = draft.trim().match(/^E([1-9]\d*):/);
    if (headerMatch) {
      return `E${headerMatch[1]}`;
    }
    // If intent has at most 1 event (or 0 events):
    if (events.length <= 1) {
      return 'E1';
    }
    return null;
  };

  const lastAttachTimeRef = useRef(0);

  const handleAttach = (file) => {
    if (!file) return;
    const now = Date.now();
    if (now - lastAttachTimeRef.current < 300) {
      return;
    }
    lastAttachTimeRef.current = now;

    const targetId = resolveTargetEventId(file);
    if (targetId) {
      onAttachImage?.(file, targetId);
    } else {
      setPendingImageFile(file);
    }
  };

  const handleFileInputChange = (event) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      handleAttach(files[0]);
    }
    event.target.value = '';
  };

  const handlePaste = (event) => {
    const items = event.clipboardData?.items;
    if (items) {
      for (let i = 0; i < items.length; i += 1) {
        if (items[i].type.startsWith('image/')) {
          const file = items[i].getAsFile();
          if (file) {
            event.preventDefault();
            event.stopPropagation();
            handleAttach(file);
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
          handleAttach(files[i]);
          return;
        }
      }
    }
  };

  const handleSelectEventTarget = (eventId) => {
    if (pendingImageFile) {
      onAttachImage?.(pendingImageFile, eventId);
      setPendingImageFile(null);
    }
  };

  // Determine dynamic submit button label.
  // Legacy /llm-rewrite, patch_events, and global_rewrite paths are removed;
  // the composer only supports NL Ask/Feedback and image attachment.
  let effectiveSubmitLabel = submitLabel;
  if (isSearching) {
    effectiveSubmitLabel = 'Searching…';
  } else if (baseIntent && submitLabel === 'Search') {
    effectiveSubmitLabel = 'Feedback';
  }

  return (
    <div
      className="kis-query-composer"
      data-testid="kis-query-composer"
    >
      {selectedContext && (
        <div className="kis-composer-context-bar" data-testid="kis-composer-context-bar">
          <span className="feedback-chip chip-context">
            Context: {selectedContext.eventId || selectedContext.resultId || 'Selected item'}
            {onClearContext && (
              <button
                type="button"
                className="kis-context-clear-btn"
                onClick={onClearContext}
                aria-label="Clear selected context"
              >
                ×
              </button>
            )}
          </span>
        </div>
      )}

      {hasStagedImages && (
        <div className="kis-composer-staged-strip" data-testid="kis-composer-staged-strip">
          {Object.entries(stagedImages || {}).flatMap(([eventId, images]) =>
            (images || []).map((img, idx) => {
              const assetId = img.asset_id || img.id;
              return (
                <div key={assetId || `${eventId}-${idx}`} className="kis-composer-staged-item">
                  <img
                    src={kisImageAssetUrl(assetId)}
                    alt={img.file_name || assetId || 'Attached reference image'}
                    className="kis-composer-staged-thumb"
                  />
                  <span className="kis-composer-staged-badge">{eventId}</span>
                  {onRemoveImage && (
                    <button
                      type="button"
                      className="kis-composer-staged-remove"
                      onClick={() => onRemoveImage(eventId, assetId)}
                      aria-label={`Remove image ${assetId} from ${eventId}`}
                      title={`Remove image from ${eventId}`}
                    >
                      ×
                    </button>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}

      {pendingImageFile && (
        <div className="kis-attach-target-chooser" role="region" aria-label="Event target chooser">
          <span className="kis-chooser-label">Attach image to:</span>
          <div className="kis-chooser-buttons">
            {events.map((e) => (
              <button
                key={e.id}
                type="button"
                className="btn-sm kis-chooser-btn"
                onClick={() => handleSelectEventTarget(e.id)}
              >
                {e.id}
              </button>
            ))}
            <button
              type="button"
              className="btn-sm btn-secondary"
              onClick={() => setPendingImageFile(null)}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {preview && (
        <div className="kis-composer-preview-bar">
          {preview.kind === 'feedback' && (
            <span className="kis-preview-badge kis-preview-feedback">
              {selectedContext?.eventId
                ? `Will refine ${selectedContext.eventId} via AI Feedback`
                : 'Will refine events via AI Feedback'}
            </span>
          )}
          {preview.error && (
            <span className="kis-preview-error" role="alert">
              ⚠️ {preview.error}
            </span>
          )}
        </div>
      )}

      <div className="kis-composer-textarea-wrapper">
        <textarea
          ref={textareaRef}
          id="event-query"
          className="input-text kis-chat-textarea"
          rows={2}
          value={draft}
          onChange={(e) => onDraftChange?.(e.target.value)}
          placeholder={placeholder}
          onFocus={onFocus}
          onBlur={onBlur}
          disabled={isSearching || disabled}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
        />
      </div>

      <div className="kis-composer-controls">
        <div className="kis-composer-left-actions">
          <label className="btn-secondary btn-sm kis-attach-label" htmlFor="kis-attach-file">
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="kis-btn-icon"
              aria-hidden="true"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
            <span>Attach image</span>
            <input
              id="kis-attach-file"
              type="file"
              accept="image/*"
              aria-label="Attach image"
              style={{ display: 'none' }}
              onChange={handleFileInputChange}
              disabled={isSearching || disabled}
            />
          </label>
        </div>

        <div className="kis-composer-right-actions">
          {renderExtraActions?.()}
          <button
            type="button"
            className="btn-primary kis-chat-send-btn"
            disabled={isSearching || disabled || (!draft.trim() && !events.length && !hasStagedImages)}
            onClick={onSubmit}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="kis-btn-icon"
              aria-hidden="true"
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <span>{effectiveSubmitLabel}</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default QueryComposer;
