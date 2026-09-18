import React from 'react';

/**
 * Lightweight confirmation and outcome modal for AVS batch submissions.
 * Prompts with aggregate counts and handles RECORDED, NOT_RECORDED, and UNKNOWN outcomes.
 */
const AvsSubmitDialog = ({
  isOpen = false,
  onClose,
  onConfirm,
  candidateCount = 0,
  uniqueVideoCount = 0,
  isSubmitting = false,
  status = 'IDLE',
  outcome = null,
  onRetryUnknown,
  onMarkUnknownRecorded,
}) => {
  if (!isOpen) return null;

  const isUnknown = status === 'UNKNOWN';
  const isRecorded = status === 'RECORDED';
  const isNotRecorded = status === 'NOT_RECORDED';

  return (
    <div className="avs-submit-dialog-backdrop" onClick={onClose}>
      <div
        className="avs-submit-dialog-content"
        role="dialog"
        aria-label="Submit AVS answers"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="avs-submit-dialog-header">
          <h2 className="avs-submit-dialog-title">Submit AVS Answers</h2>
          <button
            type="button"
            className="avs-submit-dialog-close-btn"
            onClick={onClose}
            aria-label="Close submit dialog"
          >
            ✕
          </button>
        </header>

        <div className="avs-submit-dialog-body">
          {status === 'IDLE' && (
            <p className="avs-submit-dialog-prompt">
              Submit {candidateCount} {candidateCount === 1 ? 'answer' : 'answers'} from {uniqueVideoCount} {uniqueVideoCount === 1 ? 'video' : 'videos'}?
            </p>
          )}

          {status === 'SUBMITTING' && (
            <p className="avs-submit-dialog-status loading">
              Submitting answer batch to DRES...
            </p>
          )}

          {isRecorded && (
            <div className="avs-submit-dialog-outcome success">
              <span className="avs-outcome-badge">Submitted</span>
              <p>{outcome?.message || 'DRES recorded the answer batch.'}</p>
            </div>
          )}

          {isNotRecorded && (
            <div className="avs-submit-dialog-outcome rejected">
              <span className="avs-outcome-badge error">Rejected</span>
              <p>{outcome?.message || 'Submission rejected by DRES.'}</p>
            </div>
          )}

          {isUnknown && (
            <div className="avs-submit-dialog-outcome unknown">
              <span className="avs-outcome-badge warning">Uncertain Outcome</span>
              <p>{outcome?.message || 'check DRES'}</p>
              <p className="avs-unknown-help">
                Network response was inconclusive. Please verify current competition score / log
                in the official DRES console before retrying or marking as recorded.
              </p>
            </div>
          )}
        </div>

        <footer className="avs-submit-dialog-footer">
          {status === 'IDLE' && (
            <>
              <button
                type="button"
                className="avs-bar-btn"
                onClick={onClose}
              >
                Cancel
              </button>
              <button
                type="button"
                className="avs-bar-btn avs-submit-btn"
                onClick={onConfirm}
                disabled={isSubmitting}
              >
                Confirm submit
              </button>
            </>
          )}

          {isUnknown && (
            <>
              <button
                type="button"
                className="avs-bar-btn avs-retry-btn"
                onClick={onRetryUnknown}
              >
                Retry after verification
              </button>
              <button
                type="button"
                className="avs-bar-btn avs-mark-recorded-btn"
                onClick={onMarkUnknownRecorded}
              >
                Mark recorded after verification
              </button>
            </>
          )}

          {(isRecorded || isNotRecorded) && (
            <button
              type="button"
              className="avs-bar-btn"
              onClick={onClose}
            >
              Close
            </button>
          )}
        </footer>
      </div>
    </div>
  );
};

export default AvsSubmitDialog;
