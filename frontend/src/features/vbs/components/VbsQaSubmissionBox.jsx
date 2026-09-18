import React, { useRef } from 'react';
import { useQaSubmission } from '../../submission/hooks/useQaSubmission';

/**
 * Box component inside ToolBox allowing operators to directly submit text answers
 * and verify immediate DRES verdicts for QA/VQA tasks.
 */
const VbsQaSubmissionBox = ({
  selectedTask,
  connectedUserId = '',
  onSessionRejected,
  className = '',
}) => {
  const inputRef = useRef(null);
  const {
    draft,
    setDraft,
    isSubmitting,
    error,
    lastOutcome,
    history,
    submit,
    clearHistory,
  } = useQaSubmission({
    userId: connectedUserId,
    selectedTask,
    onSessionRejected,
  });

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!draft.trim() || isSubmitting || !connectedUserId) return;
    submit();
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const isDisabled = !connectedUserId || isSubmitting;

  return (
    <div className={`vbs-qa-submission-box ${className}`.trim()}>
      <div className="vbs-qa-header">
        <span className="vbs-qa-title">QA Text Answer</span>
        {selectedTask?.taskName && (
          <span className="vbs-qa-task-badge" title={selectedTask.taskName}>
            {selectedTask.taskName}
          </span>
        )}
      </div>

      <form className="vbs-qa-form" onSubmit={handleSubmit}>
        <div className="vbs-qa-input-row">
          <input
            ref={inputRef}
            type="text"
            className="vbs-qa-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={connectedUserId ? 'Enter text answer...' : 'Connect VBS first'}
            disabled={isDisabled}
            aria-label="QA text answer"
          />
          <button
            type="submit"
            className="vbs-qa-submit-btn"
            disabled={isDisabled || !draft.trim()}
          >
            {isSubmitting ? 'Sending…' : 'Submit'}
          </button>
        </div>
        <div className="vbs-qa-helper-row">
          <span className="vbs-qa-hint">Press Enter to submit</span>
        </div>
      </form>

      {error && (
        <div className="vbs-qa-verdict-banner error" role="alert">
          <span className="vbs-qa-verdict-icon" aria-hidden="true">✕</span>
          <span className="vbs-qa-verdict-text">{error}</span>
        </div>
      )}

      {!error && lastOutcome && (
        <div
          className={`vbs-qa-verdict-banner ${lastOutcome.verdict?.toLowerCase() || lastOutcome.state?.toLowerCase() || 'info'}`}
          role="status"
          aria-live="polite"
        >
          <span className="vbs-qa-verdict-icon" aria-hidden="true">
            {lastOutcome.verdict === 'CORRECT' && '✓'}
            {lastOutcome.verdict === 'WRONG' && '✕'}
            {lastOutcome.verdict === 'INDETERMINATE' && '⏳'}
            {lastOutcome.verdict === 'UNDECIDABLE' && '?'}
            {!lastOutcome.verdict && '!'}
          </span>
          <div className="vbs-qa-verdict-content">
            <span className="vbs-qa-verdict-title">
              {lastOutcome.verdict || lastOutcome.state}
            </span>
            <span className="vbs-qa-verdict-desc">
              {lastOutcome.message || (lastOutcome.verdict === 'CORRECT' ? 'Points awarded!' : 'Try another answer.')}
            </span>
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div className="vbs-qa-history-section">
          <div className="vbs-qa-history-header">
            <span className="vbs-qa-history-title">Attempts ({history.length})</span>
            <button
              type="button"
              className="vbs-qa-history-clear-btn"
              onClick={clearHistory}
              title="Clear submission history"
            >
              Clear
            </button>
          </div>
          <ul className="vbs-qa-history-list" aria-label="QA submission history">
            {history.map((item) => (
              <li
                key={item.id}
                className={`vbs-qa-history-item ${item.verdict?.toLowerCase() || item.state?.toLowerCase() || ''}`}
              >
                <button
                  type="button"
                  className="vbs-qa-history-reuse-btn"
                  onClick={() => {
                    setDraft(item.text);
                    inputRef.current?.focus();
                  }}
                  title="Click to copy answer to input"
                >
                  <span className="vbs-qa-history-text">"{item.text}"</span>
                </button>
                <span className={`vbs-qa-chip ${item.verdict?.toLowerCase() || item.state?.toLowerCase() || ''}`}>
                  {item.verdict || item.state}
                </span>
                <span className="vbs-qa-history-time">
                  {new Date(item.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                  })}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default VbsQaSubmissionBox;
