import React from 'react';
import { kisImageAssetUrl } from '../../../api/kis';

/**
 * Present one committed or staged multimodal event with thumbnails and scoped actions.
 */
const EventCard = ({
  event,
  stagedImages = [],
  onEdit,
  onAddImage,
  onRemoveImage,
  onSelect,
  isSelected = false,
  disabled = false,
}) => {
  if (!event) return null;

  const allImages = [...(event.images || []), ...stagedImages];

  return (
    <article
      className={`kis-event-card ${isSelected ? 'kis-event-card-selected' : ''}`}
      data-testid={`kis-event-card-${event.id}`}
      onClick={() => onSelect?.(event.id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect?.(event.id);
        }
      }}
    >
      <div className="kis-event-card-header">
        <span className="kis-event-id-badge">{event.id}</span>
        {event.origin && (
          <span className={`kis-event-origin-badge kis-origin-${event.origin}`}>
            {event.origin === 'source' ? 'Source' : event.origin === 'user_override' ? 'Edited' : event.origin === 'user_added' ? 'Added' : event.origin}
          </span>
        )}
        {event.source_provenance && (
          <span
            className="kis-event-provenance-info"
            title={`Characters ${event.source_provenance.start_char}..${event.source_provenance.end_char} in query`}
          >
            [{event.source_provenance.start_char}:{event.source_provenance.end_char}]
          </span>
        )}
        {event.text && <span className="kis-event-text">{event.text}</span>}
      </div>

      {allImages.length > 0 && (
        <div className="kis-event-images">
          {allImages.map((img, idx) => {
            const assetId = img.asset_id || img.id;
            const key = assetId || `img-${idx}`;
            return (
              <div key={key} className="kis-event-image-item">
                <img
                  src={kisImageAssetUrl(assetId)}
                  alt={img.file_name || assetId || 'Event reference image'}
                  className="kis-event-thumbnail"
                />
                <button
                  type="button"
                  className="kis-event-remove-image-btn"
                  aria-label={`Remove image ${img.file_name || assetId}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveImage?.(event.id, assetId);
                  }}
                  disabled={disabled}
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
      )}

      <div className="kis-event-actions">
        <button
          type="button"
          className="kis-event-edit-btn"
          onClick={(e) => {
            e.stopPropagation();
            onEdit?.(event.id);
          }}
          disabled={disabled}
          aria-label={`Edit ${event.id}`}
        >
          <span className="kis-btn-icon" aria-hidden="true">✎</span>
          <span>Edit</span>
        </button>
        <button
          type="button"
          className="kis-event-add-image-btn"
          onClick={(e) => {
            e.stopPropagation();
            onAddImage?.(event.id);
          }}
          disabled={disabled}
          aria-label={`Add image to ${event.id}`}
        >
          <span className="kis-btn-icon" aria-hidden="true">+</span>
          <span>Add image</span>
        </button>
      </div>
    </article>
  );
};

export default EventCard;
