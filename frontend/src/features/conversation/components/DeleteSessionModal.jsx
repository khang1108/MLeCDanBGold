import React, { useEffect } from "react";

// Confirmation popup centered on screen before executing DELETE /api/v1/session/{id}
const DeleteSessionModal = ({ sessionId, onConfirm, onClose, isDeleting }) => {
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Escape" && !isDeleting) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isDeleting, onClose]);

  if (!sessionId) return null;

  return (
    <div className="modal-overlay" onClick={isDeleting ? undefined : onClose}>
      <div
        className="delete-modal-card"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="delete-modal-header">
          <div className="delete-modal-icon">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.8}
              stroke="currentColor"
              className="icon-warning"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z"
              />
            </svg>
          </div>
          <h3 className="delete-modal-title">Delete Conversation</h3>
        </div>

        <div className="delete-modal-body">
          <p>
            Are you sure you want to delete conversation{" "}
            <strong className="delete-session-highlight">{sessionId}</strong>?
          </p>
          <p className="delete-modal-warning">This action cannot be undone.</p>
        </div>

        <div className="delete-modal-actions">
          <button
            type="button"
            className="btn-utility"
            onClick={onClose}
            disabled={isDeleting}
          >
            Cancel
          </button>
          <button
            type="button"
            className="btn-danger-subtle"
            onClick={() => onConfirm(sessionId)}
            disabled={isDeleting}
          >
            {isDeleting ? "Deleting..." : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DeleteSessionModal;
