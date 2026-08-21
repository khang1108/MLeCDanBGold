import React from "react";

/**
 * Dedicated right-hand panel displaying 5 suggested queries.
 */
const QuerySuggestionsPanel = ({
  suggestions = [],
  isLoading = false,
  error = null,
  onSelectSuggestion,
  onRefresh,
}) => {
  return (
    <aside className="suggest-panel" aria-label="Suggested Queries">
      <div className="suggest-panel-header">
        <div className="suggest-panel-title-group">
          <span className="suggest-panel-icon"></span>
          <h3 className="suggest-panel-title">Query Suggestions</h3>
        </div>
        {suggestions.length > 0 && !isLoading && (
          <span className="suggest-count-badge">{suggestions.length}</span>
        )}
      </div>

      <div className="suggest-panel-body">
        {error && (
          <div className="suggest-error-box">
            <p className="suggest-error-text">{error}</p>
            {onRefresh && (
              <button
                type="button"
                className="btn-utility suggest-retry-btn"
                onClick={onRefresh}
              >
                Retry
              </button>
            )}
          </div>
        )}

        {isLoading && (
          <div className="suggest-list">
            {[1, 2, 3, 4, 5].map((idx) => (
              <div key={idx} className="suggest-card suggest-card-skeleton">
                <div className="suggest-skeleton-badge" />
                <div className="suggest-skeleton-line" />
              </div>
            ))}
          </div>
        )}

        {!isLoading && !error && suggestions.length > 0 && (
          <div className="suggest-list">
            {suggestions.slice(0, 5).map((queryText, index) => {
              const text =
                typeof queryText === "string" ? queryText : queryText.query;
              return (
                <button
                  key={`${text}-${index}`}
                  type="button"
                  className="suggest-card"
                  onClick={() => onSelectSuggestion && onSelectSuggestion(text)}
                  title={`Click to use: ${text}`}
                >
                  <span className="suggest-card-num">{index + 1}</span>
                  <span className="suggest-card-text">{text}</span>
                  <span className="suggest-card-action" aria-hidden="true">↵</span>
                </button>
              );
            })}
          </div>
        )}

        {!isLoading && !error && suggestions.length === 0 && (
          <div className="suggest-empty-state">
            <div className="suggest-empty-illustration"></div>
            <p className="suggest-empty-title">5 Query Recommendations</p>
            <p className="suggest-empty-desc">
              Click <strong>Suggest Query</strong> above to generate 5 query recommendations.
            </p>
            {onRefresh && (
              <button
                type="button"
                className="btn-utility suggest-trigger-btn"
                onClick={onRefresh}
              >
                Suggest Now
              </button>
            )}
          </div>
        )}
      </div>
    </aside>
  );
};

export default QuerySuggestionsPanel;
