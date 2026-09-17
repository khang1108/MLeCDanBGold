import React from "react";
import FrameCard from "./FrameCard";
import GifLoaderOverlay from "../../search/components/GifLoaderOverlay";

export const formatLatencySeconds = (val) => {
  if (typeof val !== 'number' || !Number.isFinite(val)) return '';
  return (val / 1000).toFixed(2) + 's';
};

// Keeps result, loading, error, warning, and welcome states in one result feature.
const FramesBox = ({
  results,
  isLoading,
  error,
  latencyMs,
  warnings = [],
  events = [],
  onFrameClick,
  onOpenSubmission,
  isSubmissionOpening = false,
  getFrameClassName,
  gridSize = 'normal',
}) => {
  const hasSearched = latencyMs !== null || error !== null;
  const hasLatency = latencyMs !== null && latencyMs !== undefined;
  const structuredLatency = typeof latencyMs === "object" && latencyMs !== null;
  const totalMs = structuredLatency ? latencyMs.total_ms : latencyMs;
  const totalLatencySec = formatLatencySeconds(totalMs);

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
      {!error && (!isLoading || results.length > 0) && hasLatency && (
        <div className="latency-banner">
          <div className="latency-summary">
            Found <span className="latency-highlight">{results.length}</span>{" "}
            frames in{" "}
            <span className="latency-highlight">{totalLatencySec}</span>
          </div>
          {structuredLatency && (
            <div className="latency-stages">
              <span className="latency-stage-item">
                Query: {formatLatencySeconds(latencyMs.query_ms)}
              </span>
              <span className="latency-stage-divider">•</span>
              <span className="latency-stage-item">
                Retrieval: {formatLatencySeconds(latencyMs.retrieval_ms)}
              </span>
              <span className="latency-stage-divider">•</span>
              <span className="latency-stage-item">
                Alignment: {formatLatencySeconds(latencyMs.alignment_ms)}
              </span>
              <span className="latency-stage-divider">•</span>
              <span className="latency-stage-item">
                Materialize: {formatLatencySeconds(latencyMs.materialization_ms)}
              </span>
            </div>
          )}
        </div>
      )}
      {warnings.length > 0 && (
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
        {(results.length > 0 || !error) &&
          (results.length ? (
            <div className={`frames-grid size-${gridSize}`}>
              {results.map((frame, index) => (
                  <FrameCard
                  key={`${frame.video_id}:${(frame.frame_ids || [frame.frame_id]).join("|")}:${index}`}
                  frame={frame}
                  events={events}
                  className={getFrameClassName?.(frame)}
                  onOpenSubmission={onOpenSubmission}
                  isSubmissionOpening={isSubmissionOpening}
                  onClick={() => onFrameClick(frame)}
                />
              ))}
            </div>
          ) : isLoading ? (
            <GifLoaderOverlay isVisible={true} />
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
