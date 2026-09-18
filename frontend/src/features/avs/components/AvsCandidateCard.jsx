import React from 'react';
import { keyframeUrl } from '../../../api/keyframes';

export const formatTimestamp = (timestampMs) => {
  if (typeof timestampMs !== 'number' || Number.isNaN(timestampMs) || timestampMs < 0) {
    return '00:00';
  }
  const totalSeconds = Math.floor(timestampMs / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
};

/**
 * Visual keyframe card in the AVS harvest grid.
 * Separates direct checkbox selection from thumbnail inspection.
 */
const AvsCandidateCard = ({
  candidate,
  selected = false,
  submitted = false,
  selectionDisabled = false,
  onToggle,
  onInspect,
  cardRef,
  tabIndex = -1,
  onKeyDown,
}) => {
  if (!candidate) return null;

  const cardClassName = [
    'avs-candidate-card',
    selected ? 'is-selected' : '',
    submitted ? 'is-submitted' : '',
  ].filter(Boolean).join(' ');

  const candidateId = candidate.candidate_id || candidate.frame_id;
  const checkboxId = `avs-select-${candidateId}`;

  return (
    <article
      ref={cardRef}
      className={cardClassName}
      data-testid={`avs-card-${candidate.candidate_id}`}
      data-candidate-id={candidate.candidate_id}
      tabIndex={tabIndex}
      onKeyDown={onKeyDown}
      onClick={() => onInspect?.(candidate)}
    >
      <div className="avs-card-header" onClick={(e) => e.stopPropagation()}>
        <label
          htmlFor={checkboxId}
          className={`avs-card-checkbox-label ${selected ? 'is-selected' : ''} ${submitted ? 'is-submitted' : ''}`}
          onClick={(e) => e.stopPropagation()}
        >
          <input
            id={checkboxId}
            type="checkbox"
            className="avs-card-checkbox"
            checked={selected}
            disabled={selectionDisabled || submitted}
            onChange={() => onToggle?.(candidate)}
          />
          <span className="avs-card-status-text">
            {submitted ? 'Submitted' : selected ? 'Selected' : 'Select'}
          </span>
        </label>
        <span className="avs-card-timestamp" title={`Timestamp: ${candidate.timestamp_ms}ms`}>
          {formatTimestamp(candidate.timestamp_ms)}
        </span>
      </div>

      <button
        type="button"
        className="avs-card-thumbnail-btn"
        onClick={(e) => {
          e.stopPropagation();
          onInspect?.(candidate);
        }}
        aria-label={`Inspect ${candidate.video_id} at ${candidate.timestamp_ms} ms`}
      >
        <img
          src={keyframeUrl(candidate.frame_id)}
          alt=""
          loading="lazy"
          className="avs-card-image"
        />
      </button>

      <div className="avs-card-meta">
        <span className="avs-card-video-id" title={candidate.video_id}>{candidate.video_id}</span>
        {Number.isSafeInteger(candidate.retrieval_rank) && (
          <span className="avs-card-rank-badge">#{candidate.retrieval_rank}</span>
        )}
      </div>
    </article>
  );
};

export default AvsCandidateCard;
