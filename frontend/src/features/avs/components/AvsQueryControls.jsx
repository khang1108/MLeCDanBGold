import React from 'react';
 import VbsTaskSelector from '../../vbs/components/VbsTaskSelector';

/**
 * Top query and task controls for the AVS workspace.
 * Includes the shared DRES task selector, query input, and search trigger.
 */
const AvsQueryControls = ({
  query = '',
  onQueryChange,
  onSearch,
  isSearching = false,
  connectedUserId = '',
  evaluations = [],
  selectedTask = null,
  setSelectedTask,
  onRequestTaskChange,
  inputRef,
}) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isSearching && query.trim()) {
      onSearch?.();
    }
  };

  return (
    <div className="avs-query-controls">
      <div className="avs-query-controls-task">
        <VbsTaskSelector
          connectedUserId={connectedUserId}
          evaluations={evaluations}
          selectedTask={selectedTask}
          onRequestChange={onRequestTaskChange || setSelectedTask}
        />
      </div>

      <form className="avs-query-form" onSubmit={handleSubmit}>
        <div className="avs-query-input-group">
          <input
            ref={inputRef}
            type="text"
            className="avs-query-input"
            value={query}
            onChange={(e) => onQueryChange?.(e.target.value)}
            placeholder="Search AVS keyframes (e.g. red sports car on highway)..."
            aria-label="AVS query"
            disabled={isSearching}
          />
          <button
            type="submit"
            className="avs-search-btn"
            disabled={isSearching || !query.trim()}
          >
            {isSearching ? 'Searching...' : 'Search AVS'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default AvsQueryControls;
