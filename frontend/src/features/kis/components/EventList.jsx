import React from 'react';
import EventCard from './EventCard';

/**
 * Render the sequence of committed events as interactive cards.
 */
const EventList = ({
  events = [],
  stagedImages = {},
  onEdit,
  onAddImage,
  onRemoveImage,
  onSelectContext,
  selectedEventId = null,
  disabled = false,
}) => {
  if (!Array.isArray(events) || events.length === 0) return null;

  return (
    <div className="kis-event-list" data-testid="kis-event-list" aria-label="Event sequence">
      {events.map((event) => (
        <EventCard
          key={event.id}
          event={event}
          stagedImages={stagedImages[event.id] || []}
          onEdit={onEdit}
          onAddImage={onAddImage}
          onRemoveImage={onRemoveImage}
          onSelect={onSelectContext}
          isSelected={selectedEventId === event.id}
          disabled={disabled}
        />
      ))}
    </div>
  );
};

export default EventList;
