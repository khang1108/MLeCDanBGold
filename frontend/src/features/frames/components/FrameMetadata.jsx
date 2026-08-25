import React from "react";
import { displayVideoId } from "../videoSource";

// Keep internal asset identity separate from BTC submission coordinates.
const FrameMetadata = ({ frame, playbackTime }) => {
  const liveFrameIdx = Number.isFinite(playbackTime) && Number.isFinite(frame.fps)
    ? Math.round(playbackTime * frame.fps)
    : frame.frame_idx;
  const liveTimestampMs = Number.isFinite(playbackTime)
    ? Math.round(playbackTime * 1000)
    : frame.timestamp_ms;

  return (
  <div className="inspector-meta-list">
    <div className="inspector-meta-item">
      <span className="meta-lbl">Internal frame ID</span>
      <span className="meta-val monospace">{frame.frame_id}</span>
    </div>
    <div className="inspector-meta-item">
      <span className="meta-lbl">Video ID</span>
      <span className="meta-val monospace">{displayVideoId(frame.video_id)}</span>
    </div>
    <div className="inspector-meta-item">
      <span className="meta-lbl">BTC frame index</span>
      <span className="meta-val monospace">{liveFrameIdx}</span>
    </div>
    {Number.isFinite(liveTimestampMs) && <div className="inspector-meta-item">
      <span className="meta-lbl">Timestamp</span>
      <span className="meta-val">{liveTimestampMs} ms</span>
    </div>}
    {Number.isFinite(frame.fps) && <div className="inspector-meta-item">
      <span className="meta-lbl">FPS</span>
      <span className="meta-val">
        {Math.abs(frame.fps - 25) <= Math.abs(frame.fps - 30) ? 25 : 30}
      </span>
    </div>}
    {Number.isFinite(frame.scores?.final) && <div className="inspector-meta-item">
      <span className="meta-lbl">Final Relevance</span>
      <span className="meta-val highlight">
        {Math.round(frame.scores.final * 100)}% ({frame.scores.final.toFixed(2)}
        )
      </span>
    </div>}
  </div>
  );
};

export default FrameMetadata;
