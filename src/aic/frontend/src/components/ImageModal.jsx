import React, { useEffect } from 'react';

const ImageModal = ({ frame, onClose }) => {
  const formattedIndex = `L21_V0001 - 01 - ${String(frame.id).padStart(2, '0')}`;

  // Close modal when Escape key is pressed
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Close Button */}
        <button className="modal-close-btn" onClick={onClose} aria-label="Close popup">
          <svg style={{ width: '20px', height: '20px' }} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Header */}
        <div className="modal-header">
          <span className="modal-title">{formattedIndex}</span>
        </div>

        {/* Image Container */}
        <div className="modal-image-container">
          <img src={frame.imageUrl} alt={formattedIndex} className="modal-image" />
        </div>

        {/* Caption Container */}
        <div className="modal-caption-container">
          <p className="modal-caption-text">{frame.caption}</p>
        </div>
      </div>
    </div>
  );
};

export default ImageModal;
