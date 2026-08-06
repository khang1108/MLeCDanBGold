import React from "react";
import FrameCard from "./FrameCard";

// Keeps result, loading, error, warning, and welcome states in one result feature.
const FramesBox = ({
  results,
  isLoading,
  error,
  latencyMs,
  warnings = [],
  onFrameClick,
  onChallengeSubmit,
  submittingFrameId,
}) => {
  const hasSearched = latencyMs !== null || error !== null;
  if (isLoading) return null;
  return (
    <section className="frames-container">
      {error && (
        <div className="error-alert">
          <div className="error-details">
            <h4 className="error-title">Search Connection Error</h4>
            <p className="error-message">{error}</p>
          </div>
        </div>
      )}
      {!error && latencyMs && (
        <div className="latency-banner">
          <div className="latency-summary">
            Found <span className="latency-highlight">{results.length}</span>{" "}
            frames in{" "}
            <span className="latency-highlight">{latencyMs.total}ms</span>
          </div>
          <div className="latency-stages">
            <span className="latency-stage-item">
              Query: {latencyMs.query_processing + latencyMs.query_encoding}ms
            </span>
            <span className="latency-stage-divider">|</span>
            <span className="latency-stage-item">
              Retrieval: {latencyMs.candidate_retrieval}ms
            </span>
            <span className="latency-stage-divider">|</span>
            <span className="latency-stage-item">
              Fusion: {latencyMs.fusion}ms
            </span>
            {latencyMs.reranking > 0 && (
              <>
                <span className="latency-stage-divider">|</span>
                <span className="latency-stage-item">
                  Rerank: {latencyMs.reranking}ms
                </span>
              </>
            )}
            <span className="latency-stage-divider">|</span>
            <span className="latency-stage-item">
              Materialize: {latencyMs.materialization}ms
            </span>
          </div>
        </div>
      )}
      {!error && warnings.length > 0 && (
        <div className="search-warning" role="status">
          <span>Server note:</span>
          <ul>
            {warnings.map((warning, index) => (
              <li key={`${warning}-${index}`}>{warning}</li>
            ))}
          </ul>
        </div>
      )}
      <div className="frames-scroll-region">
        {!error &&
          (results.length ? (
            <div className="frames-grid">
              {results.map((frame) => (
                <FrameCard
                  key={frame.frame_id}
                  frame={frame}
                  onClick={() => onFrameClick(frame)}
                />
              ))}
            </div>
          ) : (
            <div className="frames-empty-state">
              <p className="body-md frames-empty-text">
                {hasSearched
                  ? "No frames found matching your query"
                  : "Welcome to HCMAI Frame Search"}
              </p>
              <p className="caption frames-empty-subtext">
                {hasSearched
                  ? "Try adjusting your search terms or lowering the similarity threshold."
                  : "Enter a natural language question or keywords above to query the video corpus."}
              </p>
            </div>
          ))}
      </div>
    </section>
  );
};

export default FramesBox;
