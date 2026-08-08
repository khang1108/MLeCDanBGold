import React from "react";

const VqaResults = ({
  submissions,
  warnings,
  latencyMs,
  error,
  hasSearched,
  onChallengeSubmit,
  submittingFrameId,
}) => (
  <section className="frames-container vqa-results" aria-live="polite">
    {error && (
      <div className="error-alert" role="alert">
        <div className="error-details">
          <h4 className="error-title">Search Request Error</h4>
          <p className="error-message">{error}</p>
        </div>
      </div>
    )}
    {!error && hasSearched && (
      <div className="latency-banner">
        <div className="latency-summary">
          Found <span className="latency-highlight">{submissions.length}</span>{" "}
          grounded answers in{" "}
          <span className="latency-highlight">{latencyMs}ms</span>
        </div>
      </div>
    )}
    {!error && warnings.length > 0 && (
      <div className="search-warning" role="status">
        <span>Server note:</span>
        <ul>{warnings.map((warning, index) => (
          <li key={`${warning}-${index}`}>{warning}</li>
        ))}</ul>
      </div>
    )}
    {!error && submissions.length > 0 && (
      <ol className="vqa-submission-list">
        {submissions.map((submission) => (
          <li
            className="vqa-submission-card"
            key={`${submission.video_id}-${submission.frame_id}-${submission.answer}`}
          >
            <span className="vqa-rank">#{submission.rank}</span>
            <div>
              <strong>{submission.answer}</strong>
              {submission.normalized_answer
                && submission.normalized_answer !== submission.answer && (
                  <span className="vqa-normalized">
                    Normalized: {submission.normalized_answer}
                  </span>
                )}
              <span className="vqa-grounding">
                {submission.video_id}, frame {submission.frame_idx}
              </span>
            </div>
            <div className="frame-card-actions" style={{ marginLeft: "auto", marginRight: "8px" }}>
              {onChallengeSubmit && (
                <button
                  className="card-submit-btn"
                  onClick={() => onChallengeSubmit(
                    { frame_id: submission.frame_id, video_id: submission.video_id, frame_idx: submission.frame_idx },
                    submission.answer,
                  )}
                  disabled={submittingFrameId === submission.frame_id}
                  title="Submit this answer to the current mini-challenge task"
                >
                  {submittingFrameId === submission.frame_id ? "Sending…" : "Submit"}
                </button>
              )}
            </div>
            <span className="vqa-score">
              {Math.round(submission.joint_score * 100)}%
            </span>
          </li>
        ))}
      </ol>
    )}
    {!error && submissions.length === 0 && (
      <div className="frames-empty-state">
        <p className="body-md frames-empty-text">
          {hasSearched ? "No grounded answers found" : "Competition Search"}
        </p>
        <p className="caption frames-empty-subtext">
          Use /kis or /trake for retrieval, or add a question for grounded QA.
        </p>
      </div>
    )}
  </section>
);

export default VqaResults;
