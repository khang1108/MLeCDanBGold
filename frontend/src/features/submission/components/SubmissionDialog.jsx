import React, { useState } from 'react';

const outcomeMessage = (outcome) => {
  if (outcome?.message) return outcome.message;
  if (outcome?.state === 'RECORDED') return 'Recorded by DRES.';
  if (outcome?.state === 'NOT_RECORDED') return 'DRES did not record this answer.';
  if (outcome?.state === 'UNKNOWN') return 'DRES outcome is unknown. Check DRES before submitting again.';
  return '';
};

/** Edit and send one temporary answer line without retaining it after close. */
const SubmissionDialog = ({
  task,
  initialValue,
  outcome = null,
  errorMessage = '',
  isSubmitting = false,
  onChange,
  onSubmit,
  onClose,
}) => {
  const [value, setValue] = useState(initialValue ?? '');
  const statusMessage = outcomeMessage(outcome);

  return (
    <div className="submission-dialog-backdrop" onClick={onClose}>
      <section
        aria-labelledby="submission-dialog-title"
        aria-modal="true"
        className="submission-dialog"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
      >
        <header className="submission-dialog-header">
          <div>
            <p className="submission-dialog-eyebrow">{task?.task_type || 'DRES task'}</p>
            <h2 id="submission-dialog-title">Submit one answer</h2>
            <p className="submission-dialog-task">{task?.task_name || 'Current DRES task'}</p>
          </div>
          <button
            aria-label="Close submission dialog"
            className="submission-dialog-close"
            onClick={onClose}
            type="button"
          >
            ×
          </button>
        </header>

        <form
          onSubmit={(event) => {
            event.preventDefault();
            onSubmit?.(value);
          }}
        >
          <label className="submission-dialog-label" htmlFor="submission-answer-line">
            Answer
          </label>
          <textarea
            autoFocus
            id="submission-answer-line"
            onChange={(event) => {
              setValue(event.target.value);
              onChange?.(event.target.value);
            }}
            placeholder="video_id,start_ms,end_ms or a text answer"
            rows={3}
            value={value}
          />
          <p className="submission-dialog-hint">
            One line only. A three-field video range is temporal; any other line is text.
          </p>
          {errorMessage && <p className="submission-dialog-error" role="alert">{errorMessage}</p>}
          {statusMessage && (
            <p className={`submission-dialog-status ${outcome?.state?.toLowerCase() || ''}`} role="status">
              {statusMessage}
            </p>
          )}
          <footer className="submission-dialog-actions">
            <button className="submission-dialog-cancel" onClick={onClose} type="button">
              Close
            </button>
            <button className="submission-dialog-submit" disabled={isSubmitting} type="submit">
              {isSubmitting ? 'Sending…' : 'Submit'}
            </button>
          </footer>
        </form>
      </section>
    </div>
  );
};

export default SubmissionDialog;
