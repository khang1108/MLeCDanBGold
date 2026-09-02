import React from "react";
import FrameCard from "./FrameCard";

// Keeps result, loading, error, warning, and welcome states in one result feature.
const FramesBox = ({
  results,
  isLoading,
  error,
  latencyMs,
  warnings = [],
  events = [],
  onFrameClick,
  onSubmit,
  getFrameClassName,
}) => {
  const hasSearched = latencyMs !== null || error !== null;
  const hasLatency = latencyMs !== null && latencyMs !== undefined;
  const structuredLatency = typeof latencyMs === "object" && latencyMs !== null;
  const totalLatencyMs = structuredLatency ? latencyMs.total_ms : latencyMs;
  if (isLoading) return null;

  return (
    <section className="frames-container">
      {error && (
        <div className="error-alert" role="alert">
          <div className="error-details">
            <h4 className="error-title">Search Connection Error</h4>
            <p className="error-message">{error}</p>
          </div>
        </div>
      )}
      {!error && hasLatency && (
        <div className="latency-banner">
          <div className="latency-summary">
            Found <span className="latency-highlight">{results.length}</span>{" "}
            frames in{" "}
            <span className="latency-highlight">{totalLatencyMs}ms</span>
          </div>
          {structuredLatency && (
            <div className="latency-stages">
              <span className="latency-stage-item">
                Query: {latencyMs.query_ms}ms
              </span>
              <span className="latency-stage-divider">|</span>
              <span className="latency-stage-item">
                Retrieval: {latencyMs.retrieval_ms}ms
              </span>
              <span className="latency-stage-divider">|</span>
              <span className="latency-stage-item">
                Alignment: {latencyMs.alignment_ms}ms
              </span>
              <span className="latency-stage-divider">|</span>
              <span className="latency-stage-item">
                Materialize: {latencyMs.materialization_ms}ms
              </span>
            </div>
          )}
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
              {results.map((frame, index) => (
                  <FrameCard
                  key={`${frame.video_id}:${(frame.frame_ids || [frame.frame_id]).join("|")}:${index}`}
                  frame={frame}
                    events={events}
                    className={getFrameClassName?.(frame)}
                    onClick={() => onFrameClick(frame)}
                  onSubmit={onSubmit}
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
