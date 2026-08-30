import React, { useState } from "react";

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

const AlignmentAccordion = ({ events, frameIds, timestampsMs, thumbnailUrls }) => {
  const [isOpen, setIsOpen] = useState(false);
  const hasAlignment = (
    Array.isArray(events)
    && events.length > 0
    && events.length === frameIds?.length
    && events.length === timestampsMs?.length
    && events.length === thumbnailUrls?.length
  );

  if (!hasAlignment) return null;

  const toggle = (event) => {
    // The accordion lives inside a clickable card, so expanding it must not
    // also open the representative-frame inspector.
    event.stopPropagation();
    setIsOpen((open) => !open);
  };

  return (
    <div className="alignment-accordion" onClick={(event) => event.stopPropagation()}>
      <button
        type="button"
        className="alignment-accordion-toggle"
        aria-expanded={isOpen}
        onClick={toggle}
      >
        Alignment
      </button>
      {isOpen && (
        <ol className="alignment-accordion-list">
          {events.map((event, index) => (
            <li className="alignment-accordion-row" key={`${frameIds[index]}-${index}`}>
              <span className="alignment-event-label">E{index + 1}</span>
              <span className="alignment-event-text">{event}</span>
              <time className="alignment-timestamp">
                {formatTimestampMs(timestampsMs[index])}
              </time>
              <img
                className="alignment-thumbnail"
                src={thumbnailUrls[index]}
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
