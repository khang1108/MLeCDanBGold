import React from "react";
import { displayVideoId } from "../videoSource";

/**
 * Compact inspector metadata showing only canonical Video ID and Timestamp.
 * Discards ASR, OCR, Objects, Captions, and other clutter as requested.
 */
const FrameMetadata = ({ frame = {}, playbackTime }) => {
  const liveTimestampMs = Number.isFinite(playbackTime)
    ? Math.round(playbackTime * 1000)
    : frame.timestamp_ms;

  return (
    <div className="inspector-meta-container">
      <section className="inspector-meta-box" aria-label="Technical details">
        <div className="inspector-meta-grid">
          <div className="inspector-meta-cell">
            <span className="meta-cell-label">Video ID</span>
            <span className="meta-cell-value monospace">{displayVideoId(frame.video_id)}</span>
          </div>
          {Number.isFinite(liveTimestampMs) && (
            <div className="inspector-meta-cell">
              <span className="meta-cell-label">Timestamp</span>
              <span className="meta-cell-value monospace">{liveTimestampMs} ms</span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
};

export default FrameMetadata;
