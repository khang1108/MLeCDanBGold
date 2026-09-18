import React from 'react';

/**
 * FeedbackThread renders conversational chat turns between user and system,
 * displaying context chips, action summaries, and checkpoint Undo.
 */
const FeedbackThread = ({
  messages = [],
  canUndo = false,
  onUndo,
  isPending = false,
}) => {
  if (messages.length === 0 && !isPending) {
    return null;
  }

  return (
    <div className="feedback-thread" role="region" aria-label="Feedback history">
      <div className="feedback-thread-header">
        <div className="feedback-thread-title-wrap">
          <svg
            className="feedback-thread-title-icon"
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          <span className="feedback-thread-title">Feedback Thread</span>
        </div>
        {canUndo && (
          <button
            type="button"
            className="feedback-undo-button"
            onClick={onUndo}
            disabled={isPending}
            title="Revert most recent feedback turn"
          >
            ↺ Undo
          </button>
        )}
      </div>

      <div className="feedback-messages-list">
        {messages.map((msg, index) => {
          const isUser = msg.role === 'user';
          const isClarification = msg.status === 'clarification';

          return (
            <div
              key={msg.id || index}
              className={`feedback-message ${isUser ? 'feedback-message-user' : 'feedback-message-assistant'} ${
                isClarification ? 'feedback-message-clarification' : ''
              }`}
            >
              <div className="feedback-message-meta">
                <span className="feedback-role-label">
                  {isUser ? 'You' : 'Assistant'}
                </span>
                {isClarification && (
                  <span className="feedback-chip chip-clarification">Clarification</span>
                )}
                {msg.scope && (
                  <span className="feedback-chip chip-scope">
                    {msg.scope === 'all_videos' ? 'All videos' : 'Locked video'}
                  </span>
                )}
                {Array.isArray(msg.changedEventIds) && msg.changedEventIds.length > 0 && (
                  <span className="feedback-chip chip-events">
                    Events: {msg.changedEventIds.join(', ')}
                  </span>
                )}
              </div>

              {isUser && msg.context && (
                <div className="feedback-context-chip-row">
                  <span className="feedback-chip chip-context">
                    Context: {msg.context.eventId || msg.context.resultId || 'Selected'}
                  </span>
                </div>
              )}

              <div className="feedback-message-body">{msg.text}</div>
            </div>
          );
        })}

        {isPending && (
          <div className="feedback-message feedback-message-assistant feedback-pending-indicator">
            <span className="feedback-typing-dots" aria-hidden="true">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </span>
            <span>Refining retrieval...</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default FeedbackThread;
