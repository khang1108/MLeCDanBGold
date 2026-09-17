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
  getFrameAnnotation,
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
                  annotation={getFrameAnnotation?.(frame)}
                  onOpenSubmission={onOpenSubmission}
                  isSubmissionOpening={isSubmissionOpening}
                  onClick={() => onFrameClick(frame)}
                />
              ))}
            </div>
          ) : isLoading ? (
            <GifLoaderOverlay isVisible={true} />
          ) : (
            <div className={`frames-empty-state ${!hasSearched ? 'frames-empty-copyright' : ''}`}>
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
                <div className="hcmus-copyright-badge" data-testid="hcmus-copyright-badge">
                  <img
                    src="/hcmus_logo.png"
                    alt="HCMUS - Ho Chi Minh University of Science"
                    className="hcmus-copyright-logo"
                  />
                  <div className="hcmus-copyright-content">
                    <h3 className="hcmus-copyright-title">MLeCDanBGold · 2026</h3>
                    <p className="hcmus-copyright-owner">
                      Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM
                    </p>
                    <p className="hcmus-copyright-owner-en">
                      Ho Chi Minh University of Science (VNU-HCM)
                    </p>
                    <div className="hcmus-copyright-legal">
                      <p className="hcmus-copyright-statement">
                        © 2026 Team MLeCDanBGold. All Rights Reserved.
                      </p>
                      <p className="hcmus-copyright-subnote">
                        Proprietary Multimodal Video Retrieval & Reasoning System.
                        Unauthorized copying, redistribution, or reverse engineering is strictly prohibited.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
      </div>
    </section>
  );
};

export default FramesBox;
