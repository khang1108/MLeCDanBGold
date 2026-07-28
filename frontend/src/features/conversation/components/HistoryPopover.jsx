import React from 'react';

// Renders the ID-only server history endpoint without inventing conversation metadata.
const HistoryPopover = ({ sessionIds, activeId, isLoading, error, onSelect, disabled }) => (
  <div className="history-popover">
    <p className="history-popover-title">Conversations</p>
    {isLoading && <p className="history-empty">Loading…</p>}
    {error && <p className="history-error" role="alert">{error}</p>}
    {!isLoading && !error && sessionIds.length === 0 && <p className="history-empty">No conversations yet.</p>}
    {!isLoading && !error && sessionIds.map((sessionId) => (
      <button key={sessionId} className={`history-item ${activeId === sessionId ? 'active' : ''}`} onClick={() => onSelect(sessionId)} disabled={disabled}>
        <span>{sessionId}</span>
        {activeId === sessionId && <small>Active</small>}
      </button>
    ))}
  </div>
);

export default HistoryPopover;