import React from "react";

// Small overlay for local feedback draft interactions.
const FrameFeedbackActions = ({ state, onPromising, onReject, frameId }) => (
  <div className="frame-feedback-actions">
    <button
      className={`frame-feedback-btn promising ${state === "promising" ? "active" : ""}`}
      onClick={(event) => {
        event.stopPropagation();
        onPromising(frameId);
      }}
      title="Mark as promising"
      aria-label="Mark as promising"
    >
      <span aria-hidden="true">{state === "promising" ? "✓" : "✦"}</span>
    </button>
    <button
      className={`frame-feedback-btn reject ${state === "rejected" ? "active" : ""}`}
      onClick={(event) => {
        event.stopPropagation();
        onReject(frameId);
      }}
      title="Reject frame"
      aria-label="Reject frame"
    >
      <span aria-hidden="true">×</span>
    </button>
  </div>
);

export default FrameFeedbackActions;
