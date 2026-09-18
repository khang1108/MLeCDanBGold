import React, { useMemo } from 'react';
import { keyframeUrl } from '../../../api/keyframes';
import { formatTimestamp } from './AvsCandidateCard';

/**
 * Slide-out or modal review drawer for pending AVS selections.
 * Groups candidates by video ID and allows removing individual items.
 */
const AvsSelectionDrawer = ({
  isOpen = false,
  onClose,
  pending = new Map(),
  onRemove,
  onInspect,
  disabled = false,
}) => {
  const groupedByVideo = useMemo(() => {
    const groups = new Map();
    if (pending) {
      pending.forEach((candidate, id) => {
        const videoId = candidate.video_id || 'Unknown';
        if (!groups.has(videoId)) {
          groups.set(videoId, []);
        }
        groups.get(videoId).push({ ...candidate, candidate_id: candidate.candidate_id || id });
      });
    }
    return groups;
  }, [pending]);

  if (!isOpen) return null;

  return (
    <div className="avs-drawer-backdrop" onClick={onClose}>
      <div
        className="avs-drawer-content"
        role="dialog"
        aria-label="Selected AVS answers"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="avs-drawer-header">
          <h2 className="avs-drawer-title">Selected AVS answers ({pending?.size || 0})</h2>
          <button
            type="button"
            className="avs-drawer-close-btn"
            onClick={onClose}
            aria-label="Close selected answers drawer"
          >
            ✕
          </button>
        </header>

        <div className="avs-drawer-body">
          {(!pending || pending.size === 0) ? (
            <p className="avs-drawer-empty">No pending selections.</p>
          ) : (
            Array.from(groupedByVideo.entries()).map(([videoId, items]) => (
              <section key={videoId} className="avs-drawer-video-group">
                <h3 className="avs-drawer-video-title">Video: {videoId} ({items.length})</h3>
                <div className="avs-drawer-items-list">
                  {items.map((candidate) => (
                    <article
                      key={candidate.candidate_id}
                      className={`avs-drawer-item ${onInspect ? 'is-inspectable' : ''}`}
                      onClick={() => onInspect?.(candidate)}
                      role={onInspect ? 'button' : undefined}
                      tabIndex={onInspect ? 0 : undefined}
                      onKeyDown={(e) => {
                        if (onInspect && (e.key === 'Enter' || e.key === ' ')) {
                          e.preventDefault();
                          onInspect(candidate);
                        }
                      }}
                      title={onInspect ? "Click to inspect frame / video" : undefined}
                      aria-label={onInspect ? `Inspect candidate ${candidate.candidate_id}` : undefined}
                    >
                      <div className="avs-drawer-item-preview">
                        <img
                          src={keyframeUrl(candidate.frame_id)}
                          alt=""
                          className="avs-drawer-item-img"
                          loading="lazy"
                        />
                        {onInspect && (
                          <span className="avs-drawer-item-zoom-icon" aria-hidden="true">🔍</span>
                        )}
                      </div>
                      <div className="avs-drawer-item-info">
                        <span className="avs-drawer-item-id">{candidate.candidate_id}</span>
                        <span className="avs-drawer-item-ts">{formatTimestamp(candidate.timestamp_ms)}</span>
                        {onInspect && (
                          <span className="avs-drawer-item-hint">Click to inspect</span>
                        )}
                      </div>
                      <button
                        type="button"
                        className="avs-drawer-remove-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          onRemove?.(candidate.candidate_id);
                        }}
                        disabled={disabled}
                        aria-label={`Remove ${candidate.candidate_id} from selection`}
                        title="Remove from selection"
                      >
                        Remove
                      </button>
                    </article>
                  ))}
                </div>
              </section>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default AvsSelectionDrawer;
