import React from 'react';
import { keyframeUrl } from '../../../api/keyframes';

const formatSeconds = (timestampMs) => {
  if (typeof timestampMs !== 'number' || !Number.isFinite(timestampMs)) return '—';
  return `${(timestampMs / 1000).toFixed(2)}s`;
};

const HypothesisAlternatives = ({
  eventId = null,
  alternatives = [],
  activeAlternativeId = null,
  isLoading = false,
  disabled = false,
  isExhausted = false,
  onPreview,
  onUse, // Included to verify it's not called on preview click
}) => {
  if (isExhausted) {
    return null;
  }

  if (isLoading) {
    return (
      <div className="hypothesis-alternatives loading" role="status">
        <span className="alternatives-spinner">⏳</span>
        <span className="body-sm text-muted">Loading complete-path alternatives...</span>
      </div>
    );
  }

  if (!alternatives || alternatives.length === 0) {
    const targetLabel = eventId || 'selected event';
    return (
      <div className="hypothesis-alternatives empty" role="status">
        <p className="body-sm text-muted hypothesis-no-alternatives">
          No distinct alternative occurrences found for {targetLabel} under the current constraints.
        </p>
      </div>
    );
  }

  return (
    <div className="hypothesis-alternatives" aria-label="Alternative Occurrences">
      <div className="hypothesis-alternatives-header">
        <span className="hypothesis-alternatives-title">Alternative Occurrences</span>
        <span className="body-xs text-muted">
          ({alternatives.length} candidate {alternatives.length === 1 ? 'mode' : 'modes'})
        </span>
      </div>

      <div className="hypothesis-alternatives-strip" role="list">
        {alternatives.map((alt) => {
          const altId = alt.alternative_id;
          const isActive = activeAlternativeId === altId;
          const tsText = formatSeconds(alt.representative_timestamp_ms);
          const previewImg = alt.representative_frame_id
            ? keyframeUrl(alt.representative_frame_id)
            : null;

          return (
            <button
              key={altId}
              type="button"
              className={`hypothesis-alt-card ${isActive ? 'active' : ''} ${alt.is_current ? 'current' : ''}`}
              disabled={disabled}
              onClick={() => onPreview?.(alt)}
              aria-label={`Preview alternative ${altId} at ${tsText}`}
              title={`Preview alternative at ${tsText}`}
            >
              <div className="hypothesis-alt-thumb">
                {previewImg ? (
                  <img
                    src={previewImg}
                    alt={`Alternative frame ${alt.representative_frame_id}`}
                    className="hypothesis-alt-img"
                  />
                ) : (
                  <div className="hypothesis-alt-no-thumb">#{alt.representative_frame_idx}</div>
                )}
                {alt.is_current && (
                  <span className="alt-badge badge-current">Current</span>
                )}
                {isActive && (
                  <span className="alt-badge badge-preview">Previewing</span>
                )}
              </div>

              <div className="hypothesis-alt-meta">
                <span className="hypothesis-alt-time">{tsText}</span>
                <span className="hypothesis-alt-frame">#{alt.representative_frame_idx}</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default HypothesisAlternatives;
