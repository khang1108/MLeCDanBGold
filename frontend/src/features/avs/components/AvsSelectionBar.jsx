import React, { useMemo } from 'react';

/**
 * Bottom action bar showing pending candidate counts, video diversity,
 * and review/clear/submit actions.
 */
const AvsSelectionBar = ({
  pending = new Map(),
  onReview,
  onClear,
  onSubmit,
  isSubmitting = false,
  disabled = false,
}) => {
  const pendingCount = pending ? pending.size : 0;

  const uniqueVideoCount = useMemo(() => {
    if (!pending || pending.size === 0) return 0;
    const videos = new Set();
    pending.forEach((cand) => {
      if (cand?.video_id) videos.add(cand.video_id);
    });
    return videos.size;
  }, [pending]);

  return (
    <footer className="avs-selection-bar" aria-label="AVS selection bar">
      <div className="avs-selection-summary">
        <span className="avs-selection-count">
          {pendingCount} selected
        </span>
        <span className="avs-selection-separator">·</span>
        <span className="avs-selection-videos">
          {uniqueVideoCount} {uniqueVideoCount === 1 ? 'video' : 'videos'}
        </span>
      </div>

      <div className="avs-selection-actions">
        <button
          type="button"
          className="avs-bar-btn avs-review-btn"
          onClick={onReview}
          disabled={pendingCount === 0}
          aria-label="Review selections"
        >
          Review selections
        </button>

        <button
          type="button"
          className="avs-bar-btn avs-clear-btn"
          onClick={onClear}
          disabled={pendingCount === 0}
          aria-label="Clear selections"
        >
          Clear selections
        </button>

        {onSubmit && (
          <button
            type="button"
            className="avs-bar-btn avs-submit-btn"
            onClick={onSubmit}
            disabled={disabled || pendingCount === 0 || isSubmitting}
            aria-label="Submit selections"
          >
            {isSubmitting ? 'Submitting...' : `Submit ${pendingCount}`}
          </button>
        )}
      </div>
    </footer>
  );
};

export default AvsSelectionBar;
