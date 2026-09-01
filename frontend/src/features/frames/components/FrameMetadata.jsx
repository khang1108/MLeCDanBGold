import React from "react";
import { displayVideoId, normalizeSubmissionFps } from "../videoSource";

const metadataValue = (value) => (Array.isArray(value) ? value.join(", ") : value);

const objectValue = (value) => {
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") {
    return Object.entries(value)
      .map(([name, count]) => `${name}: ${count}`)
      .join(" · ");
  }
  return value;
};

// Keep internal asset identity separate from BTC submission coordinates.
const FrameMetadata = ({ frame, playbackTime }) => {
  const metadata = frame.metadata || {};
  const title = metadata.title ?? frame.title;
  const caption = metadata.caption ?? frame.caption;
  const asrText = metadata.asr ?? frame.asr ?? frame.asr_text;
  const ocrText = metadata.ocr ?? frame.ocr ?? frame.ocr_text;
  const objects = metadata.objects ?? frame.objects;
  const hasObjects = Array.isArray(objects)
    ? objects.length > 0
    : Boolean(objects && typeof objects === "object" && Object.keys(objects).length > 0);
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
    {frame.folder_id && <div className="inspector-meta-item">
      <span className="meta-lbl">Folder</span>
      <span className="meta-val monospace">{frame.folder_id}</span>
    </div>}
    {title && <div className="inspector-meta-item">
      <span className="meta-lbl">Title</span>
      <span className="meta-val">{metadataValue(title)}</span>
    </div>}
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
    {caption && <div className="inspector-meta-item inspector-meta-item-block">
      <span className="meta-lbl">Caption</span>
      <span className="meta-val">{metadataValue(caption)}</span>
    </div>}
    {asrText && <div className="inspector-meta-item inspector-meta-item-block">
      <span className="meta-lbl">ASR / Transcript</span>
      <span className="meta-val">{metadataValue(asrText)}</span>
    </div>}
    {ocrText && <div className="inspector-meta-item inspector-meta-item-block">
      <span className="meta-lbl">OCR</span>
      <span className="meta-val">{metadataValue(ocrText)}</span>
    </div>}
    {hasObjects && <div className="inspector-meta-item inspector-meta-item-block">
      <span className="meta-lbl">Objects</span>
      <span className="meta-val">{objectValue(objects)}</span>
    </div>}
  </div>
  );
};

export default FrameMetadata;
