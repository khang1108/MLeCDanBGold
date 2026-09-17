import React, { useEffect, useState } from 'react';

const TrailWindowControls = ({
  window: currentWindow = null,
  pending = false,
  onSetWindow,
  onClearWindow,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [startMs, setStartMs] = useState(currentWindow ? String(currentWindow[0]) : '');
  const [endMs, setEndMs] = useState(currentWindow ? String(currentWindow[1]) : '');

  useEffect(() => {
    if (currentWindow) {
      setStartMs(String(currentWindow[0]));
      setEndMs(String(currentWindow[1]));
    } else {
      setStartMs('');
      setEndMs('');
    }
  }, [currentWindow]);

  const numStart = Number(startMs);
  const numEnd = Number(endMs);
  const isValid = (
    startMs.trim() !== ''
    && endMs.trim() !== ''
    && Number.isSafeInteger(numStart)
    && Number.isSafeInteger(numEnd)
    && numStart >= 0
    && numStart <= numEnd
  );

  const handleApply = (e) => {
    e.preventDefault();
    if (!isValid || pending) return;
    onSetWindow?.({
      type: 'set_window',
      start_ms: numStart,
      end_ms: numEnd,
    });
  };

  const handleClear = (e) => {
    e.preventDefault();
    if (pending) return;
    onClearWindow?.();
  };

  return (
    <div className="event-trail-window-controls">
      <button
        type="button"
        className="event-trail-window-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <span>Search range</span>
        {currentWindow && (
          <span className="event-trail-badge badge-active-window">
            [{Math.round(currentWindow[0] / 1000)}s - {Math.round(currentWindow[1] / 1000)}s]
          </span>
        )}
        <span className="toggle-arrow">{isExpanded ? '▾' : '▸'}</span>
      </button>

      {isExpanded && (
        <form className="event-trail-window-form" onSubmit={handleApply}>
          <div className="event-trail-window-inputs">
            <label className="event-trail-input-label">
              <span>Start ms:</span>
              <input
                type="number"
                aria-label="Start ms"
                className="input-text event-trail-number-input"
                min="0"
                value={startMs}
                onChange={(e) => setStartMs(e.target.value)}
                placeholder="0"
                disabled={pending}
              />
            </label>
            <label className="event-trail-input-label">
              <span>End ms:</span>
              <input
                type="number"
                aria-label="End ms"
                className="input-text event-trail-number-input"
                min="0"
                value={endMs}
                onChange={(e) => setEndMs(e.target.value)}
                placeholder="60000"
                disabled={pending}
              />
            </label>
          </div>
          <div className="event-trail-window-actions">
            <button
              type="submit"
              className="btn-secondary btn-sm"
              disabled={!isValid || pending}
            >
              Apply range
            </button>
            <button
              type="button"
              className="btn-secondary btn-sm"
              disabled={pending}
              onClick={handleClear}
            >
              Clear range
            </button>
          </div>
        </form>
      )}
    </div>
  );
};

export default TrailWindowControls;
