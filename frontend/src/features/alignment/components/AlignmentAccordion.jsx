import React, { useState } from "react";
import { keyframeUrl } from "../../../api/keyframes";

export const formatTimestampMs = (timestampMs) => {
  const totalMilliseconds = Math.max(0, Math.round(timestampMs));
  const milliseconds = totalMilliseconds % 1000;
  const totalSeconds = Math.floor(totalMilliseconds / 1000);
  const seconds = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const minutes = totalMinutes % 60;
  const hours = Math.floor(totalMinutes / 60);
  const twoDigits = (value) => String(value).padStart(2, "0");
  const millisecondsText = String(milliseconds).padStart(3, "0");

  return hours > 0
    ? `${twoDigits(hours)}:${twoDigits(minutes)}:${twoDigits(seconds)}.${millisecondsText}`
    : `${twoDigits(minutes)}:${twoDigits(seconds)}.${millisecondsText}`;
};

const AlignmentAccordion = ({
  events,
  frameIds,
  timestampsMs,
  onSeek,
  collapsible = true,
  defaultOpen = false,
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(collapsible ? defaultOpen : true);
  const hasAlignment = (
    Array.isArray(events)
    && events.length > 0
    && events.length === frameIds?.length
    && events.length === timestampsMs?.length
  );

  if (!hasAlignment) return null;

  const toggle = (event) => {
    if (!collapsible) return;
    event.stopPropagation();
    setIsOpen((open) => !open);
  };

  const isExpanded = collapsible ? isOpen : true;

  return (
    <div
      className={`alignment-accordion ${!collapsible ? 'always-open' : ''} ${className}`.trim()}
      onClick={(event) => event.stopPropagation()}
    >
      {collapsible ? (
        <button
          type="button"
          className="alignment-accordion-toggle"
          aria-expanded={isExpanded}
          onClick={toggle}
        >
          Alignment
        </button>
      ) : (
        <div className="alignment-accordion-header">
          <span className="alignment-header-title">Event Alignment</span>
          <span className="alignment-header-count">{events.length} events</span>
        </div>
      )}
      {isExpanded && (
        <ol className="alignment-accordion-list">
          {events.map((event, index) => (
            <li
              className="alignment-accordion-row"
              key={`${frameIds[index]}-${index}`}
              onClick={() => onSeek?.(timestampsMs[index])}
              style={{ cursor: onSeek ? 'pointer' : 'default' }}
            >
              <span className="alignment-event-label">E{index + 1}</span>
              <span
                className="alignment-event-text"
                title={typeof event === 'string' ? event : event?.text || ''}
              >
                {typeof event === 'string' ? event : event?.text || ''}
              </span>
              {onSeek ? (
                <button
                  type="button"
                  className="alignment-timestamp"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSeek(timestampsMs[index]);
                  }}
                  title="Seek to this event"
                >
                  {formatTimestampMs(timestampsMs[index])}
                </button>
              ) : (
                <time className="alignment-timestamp">{formatTimestampMs(timestampsMs[index])}</time>
              )}
              <img
                className="alignment-thumbnail"
                src={keyframeUrl(frameIds[index])}
                alt={`Aligned frame ${frameIds[index]}`}
                loading="lazy"
              />
            </li>
          ))}
        </ol>
      )}
    </div>
  );
};

export default AlignmentAccordion;
