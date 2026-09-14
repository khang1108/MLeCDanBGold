import React from "react";
import { keyframeUrl } from "../../../api/keyframes";
import AlignmentAccordion from "../../alignment/components/AlignmentAccordion";
import { displayVideoId } from "../videoSource";
import { useOptionalAnswerWorkspace } from "../../answer-workspace/contexts/AnswerWorkspaceContext";

/** Derive result-card state only from durable answer-workspace candidates. */
export const getAnswerFrameClassName = (frame, candidates = []) => {
  if (!frame || !Array.isArray(candidates)) return '';
  const matches = candidates.filter((candidate) => candidate?.kind === 'FRAME'
    && (candidate.source_frame_id
      ? candidate.source_frame_id === frame.frame_id
      : candidate.video_id === frame.video_id && candidate.timestamp_ms === frame.timestamp_ms));
  if (matches.some((candidate) => candidate.submitted_at_ms != null
      || candidate.submitted_by_user_id != null || candidate.dres_status)) {
    return 'submitted';
  }
  return matches.length ? 'candidate' : '';
};

// Compact result card; clicking opens the inspector while controls stop propagation.
const FrameCard = ({
  frame,
  events = [],
  detail = null,
  detailStatus = 'idle',
  imageLoading = 'lazy',
  className = '',
  workspaceAction,
  onAddCandidate,
  onClick,
}) => {
  const answerWorkspace = useOptionalAnswerWorkspace();
  const displayFrame = detail ? { ...frame, ...detail } : frame;
  const frameId = displayFrame.frame_id;
  const answerFrameClassName = getAnswerFrameClassName(frame, answerWorkspace?.candidates);
  const cardClassName = [className, answerFrameClassName].filter(Boolean).join(' ');
  const previewUrl = frameId ? keyframeUrl(frameId) : null;
  const caption = displayFrame.metadata?.caption ?? displayFrame.caption;
  const hasScore = Number.isFinite(displayFrame.score);
  const hasTimestamp = Number.isFinite(displayFrame.timestamp_ms);
  const canAddFrame = workspaceAction === 'add-candidate'
    && typeof onAddCandidate === 'function'
    && typeof frame.video_id === 'string'
    && frame.video_id.trim().length > 0
    && Number.isSafeInteger(frame.timestamp_ms)
    && frame.timestamp_ms >= 0;
  return (
      <div className={`frame-card ${cardClassName}`.trim()} onClick={onClick}>
      <div className="frame-card-header">
        <span className="frame-index-text">
          {displayVideoId(displayFrame.video_id)}, {displayFrame.frame_idx}
        </span>
      </div>
      <div className="frame-image-container">
        {caption && (
          <div className="frame-tooltip">
            {caption}
            <div className="frame-tooltip-arrow" />
          </div>
        )}
        {previewUrl && detailStatus !== 'loading' ? (
          <img
            src={previewUrl}
            alt={`Frame ${frameId}`}
            className="frame-image"
            loading={imageLoading}
          />
        ) : (
          <div className="frame-image-placeholder">
            {detailStatus === 'loading'
              ? 'Loading frame…'
              : 'Preview unavailable'}
          </div>
        )}
      </div>
      <div className="frame-caption-container">
        <p className="caption frame-caption-text" title={caption || "No caption available"}>
          {caption || "No caption available"}
        </p>
      </div>
      {(hasScore || hasTimestamp) && (
        <div className="frame-card-footer">
          {hasScore && (
            <span className="frame-score-badge">
              Alignment score: {displayFrame.score.toFixed(3)}
            </span>
          )}
          {hasTimestamp && (
            <span className="frame-time-badge">{displayFrame.timestamp_ms} ms</span>
          )}
        </div>
      )}
      {canAddFrame && (
        <button
          type="button"
          className="frame-add-answer-button"
          onClick={(event) => {
            event.stopPropagation();
            onAddCandidate?.({
              kind: 'FRAME',
              videoId: frame.video_id,
              timestampMs: frame.timestamp_ms,
            });
          }}
          aria-label="Add frame to answer workspace"
        >
          ＋ Add answer
        </button>
      )}
      <AlignmentAccordion
        events={events}
        frameIds={displayFrame.frame_ids}
        timestampsMs={displayFrame.timestamps_ms}
      />
    </div>
  );
};

export default FrameCard;
