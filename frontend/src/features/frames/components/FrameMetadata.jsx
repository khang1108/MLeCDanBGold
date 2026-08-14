import React from "react";
import { displayVideoId } from "../videoSource";

// Metadata list intentionally uses only official frame identifiers.
const FrameMetadata = ({ frame }) => (
  <div className="inspector-meta-list">
    <div className="inspector-meta-item">
      <span className="meta-lbl">Frame ID</span>
      <span className="meta-val monospace">{frame.frame_id}</span>
    </div>
    <div className="inspector-meta-item">
      <span className="meta-lbl">Video ID</span>
      <span className="meta-val monospace">{displayVideoId(frame.video_id)}</span>
    </div>
    <div className="inspector-meta-item">
      <span className="meta-lbl">Frame index</span>
      <span className="meta-val monospace">{frame.frame_idx}</span>
    </div>
    {Number.isFinite(frame.timestamp_ms) && <div className="inspector-meta-item">
      <span className="meta-lbl">Timestamp</span>
      <span className="meta-val">{frame.timestamp_ms} ms</span>
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

export default FrameMetadata;
