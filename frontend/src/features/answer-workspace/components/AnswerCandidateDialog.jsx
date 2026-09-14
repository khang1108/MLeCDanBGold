import React, { useCallback, useEffect, useState } from 'react';

/** Edit one typed answer draft without mixing frame and text coordinates. */
const AnswerCandidateDialog = ({
  kind, initialValue = {}, onSave, onCancel, isEditing = false, errorMessage = '', isSaving = false,
}) => {
  const [videoId, setVideoId] = useState(initialValue.videoId || '');
  const [timestampMs, setTimestampMs] = useState(
    initialValue.timestampMs == null ? '' : String(initialValue.timestampMs),
  );
  const [text, setText] = useState(initialValue.text || '');
  const [error, setError] = useState('');
  const title = `${isEditing ? 'Edit' : 'Add'} ${kind} answer`;

  const save = useCallback((event) => {
    event?.preventDefault?.();
    setError('');
    if (kind === 'FRAME') {
      const timestamp = Number(timestampMs);
      if (!videoId.trim() || !Number.isSafeInteger(timestamp) || timestamp < 0) {
        setError('Enter a video ID and a non-negative whole-millisecond timestamp.');
        return;
      }
      onSave?.({ kind: 'FRAME', videoId: videoId.trim(), timestampMs: timestamp });
      return;
    }
    if (kind === 'TEXT' && text.trim()) {
      onSave?.({ kind: 'TEXT', text: text.trim() });
      return;
    }
    setError('Enter an answer before saving.');
  }, [kind, onSave, text, timestampMs, videoId]);

  const handleKeyDown = useCallback((event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onCancel?.();
    } else if (event.key === 'Enter') {
      event.preventDefault();
      save(event);
    }
  }, [onCancel, save]);

  useEffect(() => {
    const input = kind === 'FRAME' ? document.getElementById('answer-candidate-video-id')
      : document.getElementById('answer-candidate-text');
    input?.focus();
  }, [kind]);

  return (
    <div className="answer-dialog-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onCancel?.();
    }}>
      <form
        className="answer-candidate-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onSubmit={save}
        onKeyDown={handleKeyDown}
      >
        <header className="answer-dialog-header">
          <div>
            <span className="answer-dialog-eyebrow">Answer workspace</span>
            <h2>{title}</h2>
          </div>
          <button type="button" className="answer-icon-button" aria-label="Close answer editor" onClick={onCancel}>×</button>
        </header>
        {kind === 'FRAME' ? (
          <div className="answer-dialog-fields">
            <label>
              <span>Video ID</span>
              <input
                id="answer-candidate-video-id"
                type="text"
                value={videoId}
                onChange={(event) => setVideoId(event.target.value)}
                aria-label="Video ID"
              />
            </label>
            <label>
              <span>{isEditing ? 'timestamp_ms' : 'Timestamp (ms)'}</span>
              <input
                id="answer-candidate-timestamp-ms"
                type="number"
                min="0"
                step="1"
                value={timestampMs}
                onChange={(event) => setTimestampMs(event.target.value)}
                aria-label={isEditing ? 'timestamp_ms' : 'Timestamp (ms)'}
              />
            </label>
          </div>
        ) : (
          <label className="answer-dialog-fields answer-text-field">
            <span>Text answer</span>
            <textarea
              id="answer-candidate-text"
              value={text}
              rows={4}
              onChange={(event) => setText(event.target.value)}
              aria-label="Text answer"
            />
          </label>
        )}
        {(error || errorMessage) && <p className="answer-inline-error" role="alert">{error || errorMessage}</p>}
        <footer className="answer-dialog-actions">
          <button type="button" className="btn-secondary" onClick={onCancel}>Cancel</button>
          <button type="submit" className="btn-primary" disabled={isSaving}>
            {isSaving ? 'Saving…' : 'Save answer'}
          </button>
        </footer>
      </form>
    </div>
  );
};

export default AnswerCandidateDialog;
