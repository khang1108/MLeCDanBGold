import React from 'react';

const FrameCard = ({ index, imageUrl, caption, onClick }) => {
  // Format index as L21_V0001 - 01 - 01
  const formattedIndex = `L21_V0001 - 01 - ${String(index).padStart(2, '0')}`;

  return (
    <div className="frame-card" onClick={onClick}>
      {/* Floating Tooltip Overlay (styled via CSS hover rules on .frame-card) */}
      <div className="frame-tooltip">
        {caption}
        <div className="frame-tooltip-arrow"></div>
      </div>

      {/* Card Header showing Keyframe Code Index */}
      <div className="frame-card-header">
        <span className="frame-index-text">{formattedIndex}</span>
      </div>

      <div className="frame-image-container">
        {/* Frame Image */}
        <img
          src={imageUrl}
          alt={`Frame ${index}`}
          className="frame-image"
          loading="lazy"
        />
      </div>

      {/* Frame Caption */}
      <div className="frame-caption-container">
        <p className="caption frame-caption-text">
          {caption}
        </p>
      </div>
    </div>
  );
};

export default FrameCard;
