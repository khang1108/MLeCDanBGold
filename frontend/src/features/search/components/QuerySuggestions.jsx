import React from "react";

const QuerySuggestions = ({
  suggestions,
  isLoading,
  error,
  onSelect,
}) => {
  if (!isLoading && !error && !suggestions.length) return null;

  return (
    <section className="query-suggestions" aria-label="Query suggestions">
      {error ? (
        <p className="query-suggestions-error" role="status">
          Suggestions unavailable: {error}
        </p>
      ) : (
        <div className="query-suggestions-track" aria-busy={isLoading}>
          {isLoading
            ? Array.from({ length: 3 }, (_, index) => (
                <div
                  className="query-suggestion-card query-suggestion-skeleton"
                  key={`suggestion-skeleton-${index}`}
                  aria-hidden="true"
                />
              ))
            : suggestions.map((suggestion) => (
                <button
                  className="query-suggestion-card"
                  key={suggestion.suggestion_id}
                  type="button"
                  onClick={() => onSelect(suggestion.query)}
                  title={suggestion.query}
                >
                  <span className="query-suggestion-tags">
                    <span>{suggestion.focus}</span>
                    <span>{suggestion.language}</span>
                  </span>
                  <span className="query-suggestion-text">{suggestion.query}</span>
                </button>
              ))}
        </div>
      )}
    </section>
  );
};

export default QuerySuggestions;