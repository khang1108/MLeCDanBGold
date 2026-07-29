import React, { useEffect, useRef } from "react";

// Displays server turns in timestamp order and follows newly appended messages.
const ConversationMessages = ({ session, sessionError, isPending }) => {
  const containerRef = useRef(null);
  const turns = [...(session?.turns || [])].sort(
    (left, right) => left.created_at - right.created_at,
  );

  useEffect(() => {
    const container = containerRef.current;
    if (container)
      container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  }, [session?.session_id, turns.length, isPending]);

  return (
    <div
      ref={containerRef}
      className="conversation-messages"
      aria-live="polite"
    >
      {!session && (
        <div className="conversation-empty conversation-welcome">
          <p className="welcome-subtitle">
            {isPending
              ? "Creating conversation…"
              : "Type a query below or click '+ New' to start a session."}
          </p>
          {sessionError && (
            <p className="conversation-error" role="alert">
              {sessionError}
            </p>
          )}
        </div>
      )}
      {session && sessionError && (
        <p className="conversation-notice" role="alert">
          {sessionError}
        </p>
      )}
      {session &&
        (turns.length ? (
          turns.map((turn) => (
            <article
              key={turn.turn_id}
              className={`chat-message ${turn.sender}`}
            >
              <p>{turn.message}</p>
            </article>
          ))
        ) : (
          <div className="conversation-empty">
            <p>Start with a natural-language query, then refine it here.</p>
          </div>
        ))}
    </div>
  );
};

export default ConversationMessages;
