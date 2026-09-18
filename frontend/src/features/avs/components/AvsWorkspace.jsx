import React, { useEffect, useReducer, useRef, useState } from 'react';
import { searchAvs } from '../../../api/avs';
import {
  avsSelectionReducer,
  createInitialAvsSelectionState,
} from '../selectionState';
import AvsQueryControls from './AvsQueryControls';
import AvsHarvestGrid from './AvsHarvestGrid';

/**
 * Dedicated workspace shell for Ad-Hoc Video Search (AVS).
 * Owns text search state, basket selection state, and coordinates
 * harvest grid, selection drawer, and multi-answer submissions.
 */
const AvsWorkspace = ({
  connectedUserId = '',
  evaluations = [],
  selectedTask = null,
  setSelectedTask,
  onFrameClick,
  onInspect,
  onSessionRejected,
}) => {
  const [query, setQuery] = useState('');
  const [searchState, setSearchState] = useState({
    status: 'idle',
    results: [],
    response: null,
    error: '',
  });

  const [selectionState, dispatchSelection] = useReducer(
    avsSelectionReducer,
    undefined,
    createInitialAvsSelectionState,
  );

  const queryInputRef = useRef(null);

  const taskScopeKey = selectedTask
    ? `${selectedTask.evaluationId}:${selectedTask.taskName}`
    : null;

  useEffect(() => {
    if (taskScopeKey) {
      dispatchSelection({ type: 'BIND_SCOPE', taskScopeKey });
    }
  }, [taskScopeKey]);

  const handleSearch = async () => {
    const trimmed = query.trim();
    if (!trimmed) return;

    setSearchState((prev) => ({ ...prev, status: 'loading', error: '' }));
    try {
      const response = await searchAvs({
        query: trimmed,
        pageSize: 80,
        userId: connectedUserId || undefined,
      });
      setSearchState({
        status: 'success',
        results: response.results || [],
        response,
        error: '',
      });
    } catch (err) {
      setSearchState({
        status: 'error',
        results: [],
        response: null,
        error: err.message || 'Search failed',
      });
    }
  };

  const handleToggle = (candidate) => {
    const id = candidate.candidate_id || candidate.frame_id;
    if (selectionState.pending.has(id)) {
      dispatchSelection({ type: 'DESELECT', candidateId: id });
    } else {
      dispatchSelection({ type: 'SELECT', candidate });
    }
  };

  const handleInspect = (candidate) => {
    if (onInspect) {
      onInspect(candidate);
    } else if (onFrameClick) {
      onFrameClick(candidate);
    }
  };

  return (
    <div className="avs-workspace">
      <AvsQueryControls
        query={query}
        onQueryChange={setQuery}
        onSearch={handleSearch}
        isSearching={searchState.status === 'loading'}
        connectedUserId={connectedUserId}
        evaluations={evaluations}
        selectedTask={selectedTask}
        setSelectedTask={setSelectedTask}
        inputRef={queryInputRef}
      />

      <div className="avs-workspace-body">
        {searchState.status === 'loading' && (
          <div className="avs-status-message loading" role="status">
            Searching AVS keyframes...
          </div>
        )}

        {searchState.status === 'error' && (
          <div className="avs-status-message error" role="alert">
            {searchState.error}
          </div>
        )}

        {searchState.status === 'success' && searchState.results.length === 0 && (
          <div className="avs-status-message empty">
            No candidates found for query "{query}".
          </div>
        )}

        {searchState.results.length > 0 && (
          <>
            <div className="avs-results-summary">
              Showing {searchState.results.length} candidates from{' '}
              {searchState.response?.unique_videos ?? 0} videos (pool size:{' '}
              {searchState.response?.candidate_pool_size ?? searchState.results.length})
            </div>
            <AvsHarvestGrid
              candidates={searchState.results}
              pending={selectionState.pending}
              submitted={selectionState.submitted}
              onToggle={handleToggle}
              onInspect={handleInspect}
            />
          </>
        )}
      </div>
    </div>
  );
};

export default AvsWorkspace;
