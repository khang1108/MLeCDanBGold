import React from "react";

// Shows raw server identifiers and draft/committed feedback counts for debugging.
const ConversationDebug = ({
  session,
  requestId,
  topK,
  searchMode,
  resultCount,
  committedFeedback,
  draftFeedback,
  feedbackDirty,
}) => {
  const turns = session?.turns || [];
  const latest = (sender) =>
    [...turns].reverse().find((turn) => turn.sender === sender)?.turn_id || "—";
  return (
    <details className="conversation-debug">
      <summary>Session state</summary>
      <dl>
        <dt>Session</dt>
        <dd>{session?.session_id || "—"}</dd>
        <dt>Latest user</dt>
        <dd>{latest("user")}</dd>
        <dt>Latest AI</dt>
        <dd>{latest("ai")}</dd>
        <dt>Request</dt>
        <dd>{requestId || "—"}</dd>
        <dt>Mode</dt>
        <dd>{searchMode}</dd>
        <dt>Top K</dt>
        <dd>{topK}</dd>
        <dt>Results</dt>
        <dd>{resultCount}</dd>
        <dt>Accepted</dt>
        <dd>
          {committedFeedback.accepted_frame_ids.length} /{" "}
          {draftFeedback.accepted_frame_ids.length}
        </dd>
        <dt>Rejected</dt>
        <dd>
          {committedFeedback.rejected_frame_ids.length} /{" "}
          {draftFeedback.rejected_frame_ids.length}
        </dd>
        <dt>Feedback</dt>
        <dd>{feedbackDirty ? "Unsaved" : "Synced"}</dd>
      </dl>
    </details>
  );
};

export default ConversationDebug;
