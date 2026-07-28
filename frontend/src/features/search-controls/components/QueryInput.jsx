import React from "react";

// Lets Send submit either a text query or a dirty feedback-only draft.
const QueryInput = ({ query, setQuery, onSubmit, isSubmitting, canSubmit }) => (
  <form
    onSubmit={(event) => {
      event.preventDefault();
      if (!isSubmitting) onSubmit(query);
    }}
    className="query-form"
  >
    <div className="query-input-wrapper">
      <svg
        className="query-search-icon"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        strokeWidth={1.8}
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.603 10.602Z"
        />
      </svg>
      <input
        type="text"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Ask a question or refine the current search..."
        className="input-text query-input-field"
        disabled={isSubmitting}
      />
    </div>
    <button
      type="submit"
      className="btn-primary query-submit-btn"
      disabled={isSubmitting || !canSubmit}
    >
      {isSubmitting ? "Sending..." : "Send"}
    </button>
  </form>
);

export default QueryInput;
