import React from "react";

const QuerySuggestionsBox = ({
  suggestions = [],
  isLoading = false,
  error = null,
  onSelectSuggestion,
}) => {
  if (!isLoading && !error && (!suggestions || suggestions.length === 0)) {
    return null;
  }

  return (
    <div className="query-suggestions">
      {error && <p className="query-suggestions-error">{error}</p>}

      {isLoading && (
        <div className="query-suggestions-track">
          {[1, 2, 3].map((idx) => (
            <div key={idx} className="query-suggestion-card query-suggestion-skeleton" />
          ))}
        </div>
      )}

      {!isLoading && !error && suggestions.length > 0 && (
        <div className="query-suggestions-track">
          {suggestions.map((item) => (
            <button
              key={item.suggestion_id || item.query}
              type="button"
              className="query-suggestion-card"
              onClick={() => onSelectSuggestion && onSelectSuggestion(item.query)}
              title={`Click to use suggestion: ${item.query}`}
            >
              {item.focus && (
                <div className="query-suggestion-tags">
                  <span>{item.focus}</span>
                </div>
              )}
              <span className="query-suggestion-text">{item.query}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default QuerySuggestionsBox;
