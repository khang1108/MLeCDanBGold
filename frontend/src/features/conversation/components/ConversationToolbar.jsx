import React from "react";
import HistoryPopover from "./HistoryPopover";

// Keeps session navigation controls separate from the message list.
const ConversationToolbar = ({
  history,
  sessionId,
  isPending,
  onNew,
  onOptions,
  onToggleHistory,
  onSelectHistory,
  onDeleteHistory,
}) => (
  <div className="conversation-utility-bar">
    <div className="history-control">
      <button
        className="conversation-action-btn"
        onClick={onToggleHistory}
        disabled={isPending}
      >
        History
      </button>
      {history.isOpen && (
        <HistoryPopover
          {...history}
          activeId={sessionId}
          onSelect={onSelectHistory}
          onDeleteRequest={onDeleteHistory}
          disabled={isPending}
        />
      )}
    </div>
    <button
      className="conversation-action-btn primary"
      onClick={onNew}
      disabled={isPending}
    >
      + New
    </button>
    <button
      className="conversation-action-btn"
      onClick={onOptions}
      disabled={isPending}
    >
      Options
    </button>
    <span
      className={`conversation-status compact ${isPending ? "loading" : ""}`}
      role="status"
      aria-label={isPending ? "Working" : "Ready"}
    />
  </div>
);

export default ConversationToolbar;
