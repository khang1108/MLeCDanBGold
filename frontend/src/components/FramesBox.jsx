import React from 'react';
import FrameCard from './FrameCard';

const FramesBox = ({ results, isLoading, error, latencyMs, onFrameClick }) => {
  const skeletons = Array.from({ length: 8 });

  // Determine different render states
  const hasSearched = latencyMs !== null || error !== null;

  return (
    <section className="frames-container">
      {/* 1. Error State */}
      {error && (
        <div className="error-alert">
          <svg
            className="error-icon"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z"
            />
          </svg>
          <div className="error-details">
            <h4 className="error-title">Search Connection Error</h4>
            <p className="error-message">{error}</p>
          </div>
        </div>
      )}

      {/* 2. Latency Stats Banner */}
      {!isLoading && !error && latencyMs && (
        <div className="latency-banner">
          <div className="latency-summary">
            Found <span className="latency-highlight">{results.length}</span> frames in{" "}
            <span className="latency-highlight">{latencyMs.total}ms</span>
          </div>
          <div className="latency-stages">
            <span className="latency-stage-item">Query: {latencyMs.query_processing + latencyMs.query_encoding}ms</span>
            <span className="latency-stage-divider">|</span>
            <span className="latency-stage-item">Retrieval: {latencyMs.candidate_retrieval}ms</span>
            <span className="latency-stage-divider">|</span>
            <span className="latency-stage-item">Fusion: {latencyMs.fusion}ms</span>
            {latencyMs.reranking > 0 && (
              <>
                <span className="latency-stage-divider">|</span>
                <span className="latency-stage-item">Rerank: {latencyMs.reranking}ms</span>
              </>
            )}
            <span className="latency-stage-divider">|</span>
            <span className="latency-stage-item">Materialize: {latencyMs.materialization}ms</span>
          </div>
        </div>
      )}

      {/* 3. Grid Display: Loading skeletons vs Real cards */}
      {isLoading ? (
        <div className="frames-grid">
          {skeletons.map((_, i) => (
            <div key={i} className="frame-card skeleton">
              <div className="skeleton-header"></div>
              <div className="skeleton-image"></div>
              <div className="skeleton-caption"></div>
            </div>
          ))}
        </div>
      ) : (
        !error && (
          results.length > 0 ? (
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
            /* Empty/Welcome States */
            <div className="frames-empty-state">
              <svg
                className="frames-empty-icon"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.2}
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="m15.75 15.75-2.489-2.489m0 0a3.375 3.375 0 1 0-4.773-4.773 3.375 3.375 0 0 0 4.774 4.774ZM21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
                />
              </svg>
              {hasSearched ? (
                <>
                  <p className="body-md frames-empty-text">
                    No frames found matching your query
                  </p>
                  <p className="caption frames-empty-subtext">
                    Try adjusting your search terms or lowering the similarity threshold.
                  </p>
                </>
              ) : (
                <>
                  <p className="body-md frames-empty-text">
                    Welcome to HCMAI Frame Search
                  </p>
                  <p className="caption frames-empty-subtext">
                    Enter a natural language question or keywords above to query the video corpus.
                  </p>
                </>
              )}
            </div>
          )
        )
      )}
    </section>
  );
};

export default FramesBox;
