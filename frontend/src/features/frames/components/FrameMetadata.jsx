
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

/**
 * Structured inspection metadata with distinct, beautifully aligned cards
 * for technical coordinates, Caption, OCR, ASR transcript, and detected objects.
 */
const FrameMetadata = ({ frame = {}, playbackTime }) => {
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
    <div className="inspector-meta-container">
      {/* Technical Coordinates Grid */}
      <section className="inspector-meta-box" aria-label="Technical details">
        <div className="inspector-meta-grid">
          <div className="inspector-meta-cell">
            <span className="meta-cell-label">Video ID</span>
            <span className="meta-cell-value monospace">{displayVideoId(frame.video_id)}</span>
          </div>
          {liveFrameIdx !== undefined && liveFrameIdx !== null && (
            <div className="inspector-meta-cell">
              <span className="meta-cell-label">BTC Frame</span>
              <span className="meta-cell-value monospace">{liveFrameIdx}</span>
            </div>
          )}
          {Number.isFinite(liveTimestampMs) && (
            <div className="inspector-meta-cell">
              <span className="meta-cell-label">Timestamp</span>
              <span className="meta-cell-value monospace">{liveTimestampMs} ms</span>
            </div>
          )}
          {submissionFps !== null && (
            <div className="inspector-meta-cell">
              <span className="meta-cell-label">FPS</span>
              <span className="meta-cell-value monospace">{submissionFps}</span>
            </div>
          )}
          {Number.isFinite(frame.score) && (
            <div className="inspector-meta-cell">
              <span className="meta-cell-label">Score</span>
              <span className="meta-cell-value monospace score-highlight">
                {frame.score.toFixed(3)}
              </span>
            </div>
          )}
          {frame.folder_id && (
            <div className="inspector-meta-cell">
              <span className="meta-cell-label">Folder</span>
              <span className="meta-cell-value monospace">{frame.folder_id}</span>
            </div>
          )}
          {frame.frame_id && (
            <div className="inspector-meta-cell full-width">
              <span className="meta-cell-label">Internal ID</span>
              <span className="meta-cell-value monospace f-small">{frame.frame_id}</span>
            </div>
          )}
          {title && (
            <div className="inspector-meta-cell full-width">
              <span className="meta-cell-label">Title</span>
              <span className="meta-cell-value">{metadataValue(title)}</span>
            </div>
          )}
        </div>
      </section>

      {/* Caption Card */}
      {caption && (
        <section className="inspector-card caption-card" aria-label="Caption evidence">
          <div className="inspector-card-header">
            <span className="card-badge caption-badge">💬 Caption</span>
          </div>
          <div className="inspector-card-body caption-card-body">
            <p className="evidence-text">{metadataValue(caption)}</p>
          </div>
        </section>
      )}

      {/* OCR Card */}
      {ocrText && (
        <section className="inspector-card ocr-card" aria-label="OCR evidence">
          <div className="inspector-card-header">
            <span className="card-badge ocr-badge">🔤 OCR (Text in frame)</span>
          </div>
          <div className="inspector-card-body ocr-card-body">
            <p className="evidence-text monospace">{metadataValue(ocrText)}</p>
          </div>
        </section>
      )}

      {/* ASR Transcript Card */}
      {asrText && (
        <section className="inspector-card asr-card" aria-label="ASR transcript">
          <div className="inspector-card-header">
            <span className="card-badge asr-badge">🎙️ ASR / Speech Transcript</span>
          </div>
          <div className="inspector-card-body asr-card-body">
            <p className="evidence-text asr-text">{metadataValue(asrText)}</p>
          </div>
        </section>
      )}

      {/* Objects Card */}
      {hasObjects && (
        <section className="inspector-card objects-card" aria-label="Detected objects">
          <div className="inspector-card-header">
            <span className="card-badge objects-badge">📦 Detected Objects</span>
          </div>
          <div className="inspector-card-body objects-card-body">
            <span className="objects-value">{objectValue(objects)}</span>
          </div>
        </section>
      )}
    </div>
  );
};

export default FrameMetadata;
