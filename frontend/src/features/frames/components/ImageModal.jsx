import React, { useEffect, useState } from 'react';
import FrameMetadata from './FrameMetadata';
import ScoreBreakdown from './ScoreBreakdown';

// Full-size inspector for the currently selected result only.
const ImageModal = ({ frame, onClose }) => {
  const [copied, setCopied] = useState(false);
  const previewUrl = frame.frame_url || frame.thumbnail_url;
  useEffect(() => { const closeOnEscape = (event) => event.key === 'Escape' && onClose(); window.addEventListener('keydown', closeOnEscape); return () => window.removeEventListener('keydown', closeOnEscape); }, [onClose]);
  const copy = () => { navigator.clipboard.writeText(`${frame.video_id},${frame.frame_idx}`); setCopied(true); setTimeout(() => setCopied(false), 1200); };
  return <div className="modal-overlay" onClick={onClose}><div className="modal-card split-layout" onClick={(event) => event.stopPropagation()}><div className="modal-viewer-column">{previewUrl ? <img src={previewUrl} alt={`${frame.video_id} · frame ${frame.frame_idx}`} className="modal-viewer-image" /> : <div className="frame-image-placeholder">Preview unavailable</div>}</div><div className="modal-inspector-column"><div className="inspector-header"><span className="inspector-title">{frame.video_id} · frame {frame.frame_idx}</span><div className="inspector-header-actions"><button className={`inspector-copy-btn ${copied ? 'copied' : ''}`} onClick={copy} title="Copy official video_id,frame_idx">{copied ? '✓' : '⧉'}</button><button className="inspector-close-btn" onClick={onClose} aria-label="Close popup">×</button></div></div><div className="inspector-content"><div className="inspector-section"><span className="inspector-section-label">Caption</span><p className="inspector-caption-text">{frame.caption || 'No caption available'}</p></div><div className="inspector-section"><span className="inspector-section-label">Metadata</span><FrameMetadata frame={frame} /></div><div className="inspector-section"><span className="inspector-section-label">Retrieval Stage Scores</span><div className="inspector-scores-grid"><ScoreBreakdown scores={frame.scores} asRows /></div></div></div></div></div></div>;
};

export default ImageModal;