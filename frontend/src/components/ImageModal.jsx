import React, { useEffect, useState } from 'react';

const ImageModal = ({ frame, onClose }) => {
  const [copied, setCopied] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  const formattedIndex = `${frame.video_id} · frame ${frame.frame_idx}`;
  const previewUrl = frame.frame_url || frame.thumbnail_url;

  const handleCopy = () => {
    navigator.clipboard.writeText(`${frame.video_id},${frame.frame_idx}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  // Close modal when Escape key is pressed
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  useEffect(() => setImageFailed(false), [previewUrl]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card split-layout" onClick={(e) => e.stopPropagation()}>
        {/* Left Side: Media Viewer Column */}
        <div className="modal-viewer-column">
          {previewUrl && !imageFailed ? (
            <img
              src={previewUrl}
              alt={formattedIndex}
              className="modal-viewer-image"
              onError={() => setImageFailed(true)}
            />
          ) : (
            <div className="frame-image-placeholder">Preview unavailable</div>
          )}
        </div>

        {/* Right Side: Inspector Sidebar Column */}
        <div className="modal-inspector-column">
          {/* Header */}
          <div className="inspector-header">
            <span className="inspector-title">{formattedIndex}</span>
            <div className="inspector-header-actions">
              <button
                className={`inspector-copy-btn ${copied ? 'copied' : ''}`}
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
              <button className="inspector-close-btn" onClick={onClose} aria-label="Close popup">
                <svg style={{ width: '18px', height: '18px' }} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          <div className="inspector-content">
            {/* Caption Section */}
            <div className="inspector-section">
              <span className="inspector-section-label">Caption</span>
              <p className="inspector-caption-text">{frame.caption || 'No caption available'}</p>
            </div>

            {/* Metadata Section */}
            <div className="inspector-section">
              <span className="inspector-section-label">Metadata</span>
              <div className="inspector-meta-list">
                <div className="inspector-meta-item">
                  <span className="meta-lbl">Frame ID</span>
                  <span className="meta-val monospace">{frame.frame_id}</span>
                </div>
                <div className="inspector-meta-item">
                  <span className="meta-lbl">Video ID</span>
                  <span className="meta-val monospace">{frame.video_id}</span>
                </div>
                <div className="inspector-meta-item">
                  <span className="meta-lbl">Frame index</span>
                  <span className="meta-val monospace">{frame.frame_idx}</span>
                </div>
                <div className="inspector-meta-item">
                  <span className="meta-lbl">Timestamp</span>
                  <span className="meta-val">
                    {frame.timestamp_ms} ms
                  </span>
                </div>
                <div className="inspector-meta-item">
                  <span className="meta-lbl">Final Relevance</span>
                  <span className="meta-val highlight">
                    {Math.round(frame.scores.final * 100)}% ({frame.scores.final.toFixed(2)})
                  </span>
                </div>
              </div>
            </div>

            {/* Retrieval Stage Scores */}
            <div className="inspector-section">
              <span className="inspector-section-label">Retrieval Stage Scores</span>
              <div className="inspector-scores-grid">
                {frame.scores.visual !== null && (
                  <div className="inspector-score-row">
                    <span className="score-row-name">Visual</span>
                    <span className="score-row-val">{frame.scores.visual.toFixed(2)}</span>
                  </div>
                )}
                {frame.scores.caption !== null && (
                  <div className="inspector-score-row">
                    <span className="score-row-name">Caption</span>
                    <span className="score-row-val">{frame.scores.caption.toFixed(2)}</span>
                  </div>
                )}
                {frame.scores.ocr !== null && (
                  <div className="inspector-score-row">
                    <span className="score-row-name">OCR</span>
                    <span className="score-row-val">{frame.scores.ocr.toFixed(2)}</span>
                  </div>
                )}
                {frame.scores.asr !== null && (
                  <div className="inspector-score-row">
                    <span className="score-row-name">ASR</span>
                    <span className="score-row-val">{frame.scores.asr.toFixed(2)}</span>
                  </div>
                )}
                {frame.scores.fusion !== null && (
                  <div className="inspector-score-row">
                    <span className="score-row-name">Fusion</span>
                    <span className="score-row-val">{frame.scores.fusion.toFixed(2)}</span>
                  </div>
                )}
                {frame.scores.reranker !== null && (
                  <div className="inspector-score-row">
                    <span className="score-row-name">Reranker</span>
                    <span className="score-row-val">{frame.scores.reranker.toFixed(2)}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageModal;
