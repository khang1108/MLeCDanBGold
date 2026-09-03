import React, { useState } from 'react';

export const SqlQueryEditor = ({
  onExecute,
  isExecuting = false,
  stats = null,
  error = null,
}) => {
  const [query, setQuery] = useState('');

  const handleExecute = () => {
    if (!query.trim() || isExecuting) return;
    onExecute(query.trim());
  };

  const handleClear = () => {
    setQuery('');
  };

  const handleKeyDown = (e) => {
    // Allow Ctrl+Enter or Cmd+Enter to execute query
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleExecute();
    }
  };

  return (
    <section className="db-sql-editor" aria-label="SQL Query Console">
      <div className="db-sql-header">
        <span className="db-sql-title">SQL Query Console</span>
        <span className="db-sql-hint">Ctrl + Enter to execute</span>
      </div>

      <textarea
        className="db-sql-textarea"
        rows={3}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="SELECT * FROM query_history LIMIT 10;"
        disabled={isExecuting}
        aria-label="SQL Query"
      />

      <div className="db-sql-actions">
        <div className="db-sql-buttons">
          <button
            type="button"
            className="db-btn db-btn-primary"
            onClick={handleExecute}
            disabled={isExecuting || !query.trim()}
          >
            {isExecuting ? 'Executing...' : 'Execute SQL'}
          </button>
          <button
            type="button"
            className="db-btn"
            onClick={handleClear}
            disabled={isExecuting || !query}
          >
            Clear
          </button>
        </div>

        {stats && (
          <span className="db-sql-stats">
            {stats.is_mutation
              ? `Query executed: ${stats.rows_affected} row(s) affected in ${stats.execution_time_ms} ms.`
              : `${stats.rows_count ?? 0} row(s) returned in ${stats.execution_time_ms} ms.`}
          </span>
        )}
      </div>

      {error && (
        <div className="db-sql-error" role="alert">
          {error}
        </div>
      )}
    </section>
  );
};

export default SqlQueryEditor;
