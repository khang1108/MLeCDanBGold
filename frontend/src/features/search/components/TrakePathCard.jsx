import React from 'react';
import { keyframeUrl } from '../../../api/keyframes';
import { displayVideoId } from '../../frames/videoSource';

/** Render one backend-provided ordered TRAKE alignment path without reordering it. */
const TrakePathCard = ({ events, path, onSubmit, onFrameClick, getFrameClassName }) => (
  <article
    className="trake-path-card"
    aria-label={`TRAKE path for ${displayVideoId(path.video_id)}`}
  >
    <div className="trake-path-card-header">
      <div>
        <h3 className="trake-path-video-id">{displayVideoId(path.video_id)}</h3>
        <p className="trake-path-score">Alignment score: {path.score}</p>
      </div>
      <button
        type="button"
        className="btn-secondary trake-path-submit-btn"
        onClick={() => onSubmit?.({ ...path, frame_ids: path.frame_ids.slice() })}
      >
        Submit this path
      </button>
    </div>
    <ol className="trake-path-events">
      {path.frame_ids.map((frameId, index) => {
        const frame = {
          frame_id: frameId,
          video_id: path.video_id,
          frame_idx: path.frame_idxs[index],
          timestamp_ms: path.timestamps_ms[index],
          caption: events[index],
        };

        return (
          <li className="trake-path-event" key={`${frameId}-${index}`}>
            <button
              type="button"
              className={`trake-path-event-content ${getFrameClassName?.(frameId) || ''}`.trim()}
              onClick={() => onFrameClick?.(frame)}
              aria-label={`View event E${index + 1}: ${events[index]}`}
            >
              <span className="trake-path-event-label">E{index + 1}</span>
              <span className="trake-path-event-text">{events[index]}</span>
              <time className="trake-path-timestamp">{path.timestamps_ms[index]} ms</time>
              <img
                className="trake-path-thumbnail"
                src={keyframeUrl(frameId)}
                alt={`Frame ${frameId}`}
                loading="lazy"
              />
            </button>
          </li>
        );
      })}
    </ol>
  </article>
);

export default TrakePathCard;
