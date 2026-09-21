import React from 'react';

const formatSeconds = (timestampMs) => {
  if (typeof timestampMs !== 'number' || !Number.isFinite(timestampMs)) return '—';
  return `${(timestampMs / 1000).toFixed(2)}s`;
};

const HypothesisPathPreview = ({
  activePath = [],
  previewAlternative = null,
  pending = false,
  onUseAlternative,
  onClearPreview,
}) => {
  if (!previewAlternative || !previewAlternative.path) {
    return null;
  }

  const focusedEventId = previewAlternative.event_id;
  const previewPath = previewAlternative.path;

  // Build active map by event_id for fast diffing
  const activeMap = new Map();
  activePath.forEach((c) => {
    activeMap.set(c.event_id, c);
  });

  return (
    <div className="hypothesis-path-preview" data-testid="hypothesis-path-preview">
      <div className="hypothesis-path-preview-header">
        <div className="preview-badge-group">
          <span className="hypothesis-badge badge-preview">Previewing Complete Path</span>
          <span className="body-xs text-muted">
            Focus event: <strong>{focusedEventId}</strong>
          </span>
        </div>
        <div className="preview-header-actions">
          <button
            type="button"
            className="btn-primary btn-sm hypothesis-use-alt-btn"
            disabled={pending}
            onClick={() =>
              onUseAlternative?.(
                focusedEventId,
                previewAlternative.alternative_id
              )
            }
          >
            Use this occurrence
          </button>
          <button
            type="button"
            className="btn-secondary btn-sm hypothesis-cancel-preview-btn"
            disabled={pending}
            onClick={() => onClearPreview?.()}
          >
            Clear preview
          </button>
        </div>
      </div>

      <div className="hypothesis-path-preview-events">
        {previewPath.map((cand) => {
          const isFocused = cand.event_id === focusedEventId;
          const currentCand = activeMap.get(cand.event_id);
          const isChanged =
            !currentCand ||
            currentCand.frame_id !== cand.frame_id ||
            currentCand.timestamp_ms !== cand.timestamp_ms;

          let statusTag = null;
          if (isFocused) {
            statusTag = <span className="event-tag badge-focused">Selected</span>;
          } else if (isChanged) {
            statusTag = (
              <span className="event-tag badge-adjusted" data-testid="adjusted-event">
                adjusted to maintain order
              </span>
            );
          } else {
            statusTag = <span className="event-tag badge-unchanged">Unchanged</span>;
          }

          return (
            <div
              key={cand.event_id}
              className={`hypothesis-path-preview-item ${isFocused ? 'focused' : ''} ${isChanged ? 'adjusted' : ''}`}
            >
              <div className="preview-item-left">
                <span className="event-id">{cand.event_id}</span>
                {statusTag}
              </div>
              <div className="preview-item-coords">
                <span className="frame-num">#{cand.frame_idx}</span>
                <span className="time-val">
                  {currentCand && isChanged && !isFocused
                    ? `${formatSeconds(currentCand.timestamp_ms)} → `
                    : ''}
                  {formatSeconds(cand.timestamp_ms)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default HypothesisPathPreview;
