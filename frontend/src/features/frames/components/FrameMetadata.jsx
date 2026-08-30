import React from "react";
import { displayVideoId, normalizeSubmissionFps } from "../videoSource";

const metadataValue = (value) => (Array.isArray(value) ? value.join(", ") : value);

// Keep internal asset identity separate from BTC submission coordinates.
const FrameMetadata = ({ frame, playbackTime }) => {
  const metadata = frame.metadata || {};
  const submissionFps = normalizeSubmissionFps(frame.fps);
  const liveFrameIdx = Number.isFinite(playbackTime) && submissionFps !== null
    ? Math.round(playbackTime * submissionFps)
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
    {submissionFps !== null && <div className="inspector-meta-item">
      <span className="meta-lbl">FPS</span>
      <span className="meta-val">
        {submissionFps}
      </span>
    </div>}
    {Number.isFinite(frame.score) && <div className="inspector-meta-item">
      <span className="meta-lbl">Alignment score</span>
      <span className="meta-val highlight">
        {frame.score.toFixed(3)}
      </span>
    </div>}
    {metadata.title && <div className="inspector-meta-item">
      <span className="meta-lbl">Title</span>
      <span className="meta-val">{metadataValue(metadata.title)}</span>
    </div>}
    {metadata.caption && <div className="inspector-meta-item">
      <span className="meta-lbl">Caption</span>
      <span className="meta-val">{metadataValue(metadata.caption)}</span>
    </div>}
    {metadata.ocr && <div className="inspector-meta-item">
      <span className="meta-lbl">OCR</span>
      <span className="meta-val">{metadataValue(metadata.ocr)}</span>
    </div>}
    {metadata.objects?.length > 0 && <div className="inspector-meta-item">
      <span className="meta-lbl">Objects</span>
      <span className="meta-val">{metadataValue(metadata.objects)}</span>
    </div>}
    {metadata.asr && <div className="inspector-meta-item">
      <span className="meta-lbl">ASR</span>
      <span className="meta-val">{metadataValue(metadata.asr)}</span>
    </div>}
  </div>
  );
};

export default FrameMetadata;
