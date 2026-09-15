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
  disabled = false,
}) => {
  if (!event) return null;

  const allImages = [...(event.images || []), ...stagedImages];

  return (
    <article className="kis-event-card" data-testid={`kis-event-card-${event.id}`}>
      <div className="kis-event-card-header">
        <span className="kis-event-id-badge">{event.id}</span>
        {event.text && <span className="kis-event-text">{event.text}</span>}
        <div className="kis-event-actions">
          <button
            type="button"
            className="btn-sm kis-event-edit-btn"
            onClick={() => onEdit?.(event.id)}
            disabled={disabled}
            aria-label={`Edit ${event.id}`}
          >
            Edit
          </button>
          <button
            type="button"
            className="btn-sm kis-event-add-image-btn"
            onClick={() => onAddImage?.(event.id)}
            disabled={disabled}
            aria-label={`Add image to ${event.id}`}
          >
            Add image
          </button>
        </div>
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
                  onClick={() => onRemoveImage?.(event.id, assetId)}
                  disabled={disabled}
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
      )}
    </article>
  );
};

export default EventCard;
