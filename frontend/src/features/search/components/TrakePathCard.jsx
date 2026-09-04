import React from 'react';
import FrameCard from '../../frames/components/FrameCard';
import { displayVideoId } from '../../frames/videoSource';

/** Render one ordered TRAKE path with the established result-card grid treatment. */
const TrakePathCard = ({ events, path, onSubmit, onFrameClick, getFrameClassName }) => {
  const videoId = displayVideoId(path.video_id);

  return (
    <article className="trake-video-group" aria-label={`TRAKE frames for ${videoId}`}>
      <h3 className="trake-video-heading">
        {videoId}
        <button
          type="button"
          className="btn-secondary trake-path-submit-btn"
          onClick={() => onSubmit?.({ ...path, frame_ids: path.frame_ids.slice() })}
          title="Submit this TRAKE path"
          aria-label={`Submit TRAKE path for ${videoId}`}
        >
          ⬆
        </button>
      </h3>
      <div className="frames-grid">
        {path.frame_ids.map((frameId, index) => {
          const frame = {
            frame_id: frameId,
            video_id: path.video_id,
            frame_idx: path.frame_idxs[index],
            timestamp_ms: path.timestamps_ms[index],
            caption: events[index],
          };

          return (
            <FrameCard
              key={`${frameId}-${index}`}
              frame={frame}
              className={getFrameClassName?.(frameId) || ''}
              onClick={() => onFrameClick?.(frame)}
            />
          );
        })}
      </div>
    </article>
  );
};

export default TrakePathCard;
