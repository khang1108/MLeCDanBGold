import React, { useEffect, useReducer, useRef, useState } from 'react';
import { searchAvs } from '../../../api/avs';
import { getCurrentDresTask } from '../../../api/submissions';
import {
  avsSelectionReducer,
  createInitialAvsSelectionState,
} from '../selectionState';
import AvsQueryControls from './AvsQueryControls';
import AvsHarvestGrid from './AvsHarvestGrid';
import AvsSelectionBar from './AvsSelectionBar';
import AvsSelectionDrawer from './AvsSelectionDrawer';

/**
 * Dedicated workspace shell for Ad-Hoc Video Search (AVS).
 * Owns text search state, basket selection state, and coordinates
 * harvest grid, selection drawer, live task-scope binding, and multi-answer submissions.
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

  const [scopeConflict, setScopeConflict] = useState(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const queryInputRef = useRef(null);
  const pendingTaskSwitchRef = useRef(null);

  // Live scope resolution via getCurrentDresTask
  useEffect(() => {
    let isCancelled = false;

    const resolveScope = async () => {
      try {
        const dresTask = await getCurrentDresTask(connectedUserId, {
          evaluationId: selectedTask?.evaluationId,
          taskName: selectedTask?.taskName,
        });
        if (isCancelled) return;

        const liveScopeKey = dresTask?.task_scope_key || (selectedTask ? `${selectedTask.evaluationId}:${selectedTask.taskName}` : null);
        if (!liveScopeKey) return;

        if (pendingTaskSwitchRef.current) {
          pendingTaskSwitchRef.current = null;
          dispatchSelection({ type: 'RESET_FOR_SCOPE', taskScopeKey: liveScopeKey });
          setScopeConflict(null);
        } else if (selectionState.taskScopeKey === null) {
          dispatchSelection({ type: 'BIND_SCOPE', taskScopeKey: liveScopeKey });
          setScopeConflict(null);
        } else if (selectionState.taskScopeKey !== liveScopeKey) {
          if (selectionState.pending.size > 0) {
            setScopeConflict(
              `Task scope changed on the server (${liveScopeKey} != ${selectionState.taskScopeKey}). Selections are locked.`
            );
          } else {
            dispatchSelection({ type: 'RESET_FOR_SCOPE', taskScopeKey: liveScopeKey });
            setScopeConflict(null);
          }
        } else {
          setScopeConflict(null);
        }
      } catch (err) {
        if (!isCancelled && err?.status === 401) {
          onSessionRejected?.();
        }
      }
    };

    resolveScope();

    return () => {
      isCancelled = true;
    };
  }, [connectedUserId, selectedTask?.evaluationId, selectedTask?.taskName, selectionState.taskScopeKey, selectionState.pending.size, onSessionRejected]);

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

  const handleTaskChangeRequest = (nextTask) => {
    if (!nextTask) {
      setSelectedTask?.(null);
      return;
    }
    if (selectionState.pending.size === 0) {
      setSelectedTask?.(nextTask);
      return;
    }
    const count = selectionState.pending.size;
    const confirmed = window.confirm(
      `Discard ${count} pending AVS selection${count === 1 ? '' : 's'} and switch task?`
    );
    if (confirmed) {
      pendingTaskSwitchRef.current = nextTask;
      setSelectedTask?.(nextTask);
    }
  };

  const handleClearPending = () => {
    const count = selectionState.pending.size;
    if (count === 0) return;
    const confirmed = window.confirm(
      `Clear ${count} pending AVS selection${count === 1 ? '' : 's'}?`
    );
    if (confirmed) {
      dispatchSelection({ type: 'CLEAR_PENDING' });
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

  const handleRemoveCandidate = (candidateId) => {
    dispatchSelection({ type: 'DESELECT', candidateId });
  };

  const handleInspect = (candidate) => {
    if (onInspect) {
      onInspect(candidate);
    } else if (onFrameClick) {
      onFrameClick(candidate);
    }
  };

  const isSelectionDisabled = Boolean(scopeConflict) || Boolean(selectionState.unknownBatch);

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
        onRequestTaskChange={handleTaskChangeRequest}
        inputRef={queryInputRef}
      />

      <div className="avs-workspace-body">
        {scopeConflict && (
          <div className="avs-status-message error" role="alert">
            {scopeConflict}
          </div>
        )}

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
              selectionDisabled={isSelectionDisabled}
              onToggle={handleToggle}
              onInspect={handleInspect}
            />
          </>
        )}
      </div>

      <AvsSelectionBar
        pending={selectionState.pending}
        onReview={() => setIsDrawerOpen(true)}
        onClear={handleClearPending}
        disabled={isSelectionDisabled}
      />

      <AvsSelectionDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        pending={selectionState.pending}
        onRemove={handleRemoveCandidate}
        disabled={isSelectionDisabled}
      />
    </div>
  );
};

export default AvsWorkspace;
