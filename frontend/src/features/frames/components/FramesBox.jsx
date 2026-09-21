import React, { useCallback } from "react";
import FrameCard from "./FrameCard";
import GifLoaderOverlay from "../../search/components/GifLoaderOverlay";
import HcmusWatermarkBadge from "./HcmusWatermarkBadge";

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
  eventTrail = null,
  eventTrailContext = null,
}) => {
  const activeTrailSession = eventTrail?.session;
  const isTrailPending = Boolean(eventTrail?.pending);

  const handleStartTrail = useCallback(async (resultItem) => {
    if (!eventTrailContext?.snapshotId || !resultItem?.result_id) return null;
    const ctx = {
      snapshotId: eventTrailContext.snapshotId,
      resultId: resultItem.result_id,
      kisRevision: eventTrailContext.kisRevision,
      events: eventTrailContext.events,
      searchSessionId: eventTrailContext.searchSessionId,
    };
    return await eventTrail?.open?.(ctx);
  }, [eventTrailContext, eventTrail]);

  const handleClearAnchor = useCallback(async (resultItem, eventLabel) => {
    if (!resultItem || !eventLabel) return;
    await eventTrail?.act?.({ type: 'clear_anchor', event_id: eventLabel });
  }, [eventTrail]);

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
      <div className={`frames-scroll-region ${isLoading ? 'whip-cursor-mode' : ''}`}>
        {(results.length > 0 || !error) &&
          (results.length ? (
            <div className={`frames-grid size-${gridSize}`}>
              {results.map((resultItem, index) => {
                const isRowTrailActive = Boolean(
                  resultItem?.result_id
                  && activeTrailSession?.result_id
                  && activeTrailSession.result_id === resultItem.result_id,
                );

                const activePath = isRowTrailActive
                  ? (Array.isArray(activeTrailSession?.path) && activeTrailSession.path.length > 0
                      ? activeTrailSession.path
                      : (Array.isArray(activeTrailSession?.last_valid_path) && activeTrailSession.last_valid_path.length > 0
                          ? activeTrailSession.last_valid_path
                          : null))
                  : null;

                const frameIds = activePath && activePath.length > 0
                  ? activePath.map((c) => c.frame_id)
                  : (Array.isArray(resultItem.frame_ids) && resultItem.frame_ids.length > 0
                    ? resultItem.frame_ids
                    : [resultItem.frame_id]);
                const timestampsMs = activePath && activePath.length > 0
                  ? activePath.map((c) => c.timestamp_ms)
                  : (Array.isArray(resultItem.timestamps_ms) && resultItem.timestamps_ms.length > 0
                    ? resultItem.timestamps_ms
                    : [resultItem.timestamp_ms]);
                const hasMultipleEvents = frameIds.length > 1;


                const rejectedCounts = isRowTrailActive && activeTrailSession ? (activeTrailSession.rejected_counts || {}) : {};
                const canStartTrail = Boolean(eventTrailContext?.snapshotId && resultItem?.result_id);

                return (
                  <div
                    key={`${resultItem.video_id}:${resultItem.result_id || frameIds.join("|")}:${index}`}
                    className={`frames-result-row ${isRowTrailActive ? 'trail-row-active' : ''}`}
                  >
                    {(isRowTrailActive || canStartTrail) && (
                      <div className="frames-row-header">
                        <div className="frames-row-trail-bar">
                        {isRowTrailActive && activeTrailSession ? (
                          <div className="frames-row-trail-active-group">
                            <span className="badge-trail-active">
                              ⚡ Explorer Rev {activeTrailSession.trail_revision}
                            </span>
                            {activeTrailSession.status === 'exhausted' && (
                              <span className="badge-trail-exhausted">Exhausted</span>
                            )}
                            <button
                              type="button"
                              className="btn-trail-row-action btn-trail-undo"
                              onClick={() => eventTrail.undo()}
                              disabled={isTrailPending}
                              title="Undo last Hypothesis Explorer action"
                            >
                              ↺ Undo
                            </button>
                            <button
                              type="button"
                              className="btn-trail-row-action btn-trail-exit"
                              onClick={() => eventTrail.close()}
                              disabled={isTrailPending}
                              title="Exit Hypothesis Explorer"
                            >
                              Exit Explorer
                            </button>
                          </div>
                        ) : canStartTrail ? (
                          <button
                            type="button"
                            className="btn-trail-row-start"
                            onClick={() => handleStartTrail(resultItem)}
                            disabled={isTrailPending}
                            title="Explore timeline alignment with Hypothesis Explorer"
                          >
                            ⚡ Hypothesis Explorer
                          </button>
                        ) : null}
                      </div>
                    </div>
                  )}

                    <div className="frames-row-track">
                      {frameIds.map((fId, eventIndex) => {
                        const eventLabel = hasMultipleEvents ? `E${eventIndex + 1}` : 'E1';

                        const rejectedCount = rejectedCounts[eventLabel] || 0;
                        const timestampMs = timestampsMs[eventIndex] ?? resultItem.timestamp_ms;
                        const eventItem = events?.[eventIndex];
                        const eventText = typeof eventItem === 'string'
                          ? eventItem
                          : (eventItem?.text || eventItem?.canonical_text || null);
                          
                        const trailCandidate = activePath?.[eventIndex];
                        const eventFrame = {
                          ...resultItem,
                          frame_id: fId,
                          frame_ids: frameIds,
                          timestamps_ms: timestampsMs,
                          timestamp_ms: timestampMs,
                          frame_idx: trailCandidate?.frame_idx ?? resultItem.frame_idx,
                          event_index: eventIndex,
                          eventIndex,
                          event_label: eventLabel,
                          eventLabel,
                          event_text: eventText,
                          eventText,
                        };
                        return (
                          <FrameCard
                            key={`${fId}-${eventIndex}-${activeTrailSession?.trail_revision || 0}`}
                            frame={eventFrame}
                            eventLabel={hasMultipleEvents ? eventLabel : null}
                            events={events}
                            className={getFrameClassName?.(eventFrame)}
                            annotation={getFrameAnnotation?.(eventFrame)}
                            onOpenSubmission={onOpenSubmission}
                            isSubmissionOpening={isSubmissionOpening}
                            onClick={() => onFrameClick(eventFrame)}
                            showTrailActions={isRowTrailActive || canStartTrail}
                            isTrailPending={isTrailPending}
                            onClearAnchor={() => handleClearAnchor(resultItem, eventLabel)}
                          />
                        );
                      })}
                    </div>
                  </div>
                );
              })}
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
                <HcmusWatermarkBadge />
              )}
            </div>
          ))}
      </div>
    </section>
  );
};

export default FramesBox;
