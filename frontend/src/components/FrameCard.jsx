import React, { useState } from 'react';

const FrameCard = ({ frame, onClick }) => {
  const [copied, setCopied] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);

  const formattedIndex = `${frame.video_id} · frame ${frame.frame_idx}`;
  const previewUrl = frame.thumbnail_url || frame.frame_url;

  const handleCopy = (e) => {
    e.stopPropagation(); // prevent modal trigger
    navigator.clipboard.writeText(`${frame.video_id},${frame.frame_idx}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className="frame-card" onClick={onClick}>
      {/* 1. Floating Caption Tooltip Overlay (on Card Hover) */}
      <div className="frame-tooltip">
        {frame.caption}
        <div className="frame-tooltip-arrow"></div>
      </div>

      {/* 3. Card Header showing Keyframe Code Index */}
      <div className="frame-card-header">
        <span className="frame-index-text">{formattedIndex}</span>
        <button
          className={`card-copy-btn ${copied ? 'copied' : ''}`}
          onClick={handleCopy}
          title="Copy official video_id,frame_idx"
          aria-label="Copy official video and frame identifiers"
        >
          {copied ? (
            <svg className="copy-icon check" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
          ) : (
            <svg className="copy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          )}
        </button>
      </div>

      {/* 4. Image Container */}
      <div className="frame-image-container">
        {previewUrl && !imageFailed ? (
          <img
            src={previewUrl}
            alt={`Frame ${frame.frame_id}`}
            className="frame-image"
            loading="lazy"
            onError={() => setImageFailed(true)}
          />
        ) : (
          <div className="frame-image-placeholder">Preview unavailable</div>
        )}
      </div>

      {/* 5. Frame Caption Truncated */}
      <div className="frame-caption-container">
        <p className="caption frame-caption-text">
          {frame.caption || 'No caption available'}
        </p>
      </div>

      {/* 6. Card Footer with Score & Timestamp Meta info */}
      <div className="frame-card-footer">
        {/* Hoverable Score Badge */}
        <div className="frame-score-badge-wrapper">
          <span className="frame-score-badge">
            Score: {Math.round(frame.scores.final * 100)}%
          </span>
          <div className="score-tooltip">
            <div className="score-tooltip-title">Score Details</div>
            <table className="score-tooltip-table">
              <tbody>
                {frame.scores.visual !== null && (
                  <tr>
                    <td className="score-name">Visual:</td>
                    <td className="score-value">{frame.scores.visual.toFixed(2)}</td>
                  </tr>
                )}
                {frame.scores.caption !== null && (
                  <tr>
                    <td className="score-name">Caption:</td>
                    <td className="score-value">{frame.scores.caption.toFixed(2)}</td>
                  </tr>
                )}
                {frame.scores.ocr !== null && (
                  <tr>
                    <td className="score-name">OCR:</td>
                    <td className="score-value">{frame.scores.ocr.toFixed(2)}</td>
                  </tr>
                )}
                {frame.scores.asr !== null && (
                  <tr>
                    <td className="score-name">ASR:</td>
                    <td className="score-value">{frame.scores.asr.toFixed(2)}</td>
                  </tr>
                )}
                {frame.scores.fusion !== null && (
                  <tr>
                    <td className="score-name">Fusion:</td>
                    <td className="score-value">{frame.scores.fusion.toFixed(2)}</td>
                  </tr>
                )}
                {frame.scores.reranker !== null && (
                  <tr>
                    <td className="score-name">Rerank:</td>
                    <td className="score-value">{frame.scores.reranker.toFixed(2)}</td>
                  </tr>
                )}
                <tr className="score-tooltip-divider">
                  <td colSpan="2"></td>
                </tr>
                <tr className="score-tooltip-highlight">
                  <td className="score-name">Final:</td>
                  <td className="score-value">{frame.scores.final.toFixed(2)}</td>
                </tr>
              </tbody>
            </table>
            <div className="score-tooltip-arrow"></div>
          </div>
        </div>

        {/* Time Stamp Tag */}
        <span className="frame-time-badge">
          {frame.timestamp_ms} ms
        </span>
      </div>
    </div>
  );
};

export default FrameCard;
