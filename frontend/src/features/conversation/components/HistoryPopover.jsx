import React from "react";

// Renders the ID-only server history endpoint without inventing conversation metadata.
const HistoryPopover = ({
  sessionIds,
  activeId,
  isLoading,
  error,
  onSelect,
  onDeleteRequest,
  disabled,
}) => (
  <div className="history-popover">
    <p className="history-popover-title">Conversations</p>
    {isLoading && <p className="history-empty">Loading…</p>}
    {error && (
      <p className="history-error" role="alert">
        {error}
      </p>
    )}
    {!isLoading && !error && sessionIds.length === 0 && (
      <p className="history-empty">No conversations yet.</p>
    )}
    {!isLoading &&
      !error &&
      sessionIds.map((sessionId) => (
        <div
          key={sessionId}
          className={`history-item ${activeId === sessionId ? "active" : ""}`}
          onClick={() => onSelect(sessionId)}
          role="button"
          tabIndex={0}
        >
          <span className="history-item-label">{sessionId}</span>
          {activeId === sessionId && (
            <small className="history-active-badge">Active</small>
          )}
          <button
            type="button"
            className="history-delete-btn"
            title="Delete conversation"
            onClick={(event) => {
              event.stopPropagation();
              onDeleteRequest?.(sessionId);
            }}
            disabled={disabled}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.8}
              stroke="currentColor"
              className="trash-icon"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"
              />
            </svg>
          </button>
        </div>
      ))}
  </div>
);

export default HistoryPopover;
