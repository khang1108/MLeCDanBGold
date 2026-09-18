/**
 * Pure selection reducer and state helpers for AVS workspace basket.
 *
 * Tracks pending candidates for multi-answer batch submission and already
 * submitted candidates to prevent duplicate submission within a task scope.
 */

export const createInitialAvsSelectionState = (taskScopeKey = null) => ({
  taskScopeKey,
  pending: new Map(),
  submitted: new Map(),
  unknownBatch: null,
});

export const candidateToTemporalAnswer = (candidate) => {
  if (!candidate || typeof candidate !== 'object') {
    throw new Error('Invalid candidate for temporal answer');
  }
  return {
    kind: 'TEMPORAL',
    video_id: candidate.video_id,
    start_ms: candidate.timestamp_ms,
    end_ms: candidate.timestamp_ms,
  };
};

const isValidCandidate = (c) => {
  if (!c || typeof c !== 'object') return false;
  const id = c.candidate_id || c.frame_id;
  if (typeof id !== 'string' || !id.trim()) return false;
  if (typeof c.video_id !== 'string' || !c.video_id.trim()) return false;
  if (typeof c.timestamp_ms !== 'number' || !Number.isSafeInteger(c.timestamp_ms) || c.timestamp_ms < 0) return false;
  return true;
};

export const avsSelectionReducer = (state, action) => {
  if (!action || typeof action !== 'object') return state;

  switch (action.type) {
    case 'BIND_SCOPE': {
      if (state.taskScopeKey === action.taskScopeKey) {
        return state;
      }
      return {
        ...state,
        taskScopeKey: action.taskScopeKey ?? null,
      };
    }

    case 'SELECT': {
      const candidate = action.candidate;
      if (!isValidCandidate(candidate)) {
        return state;
      }
      const id = candidate.candidate_id || candidate.frame_id;
      if (state.submitted.has(id)) {
        return state;
      }
      if (state.pending.has(id)) {
        return state;
      }
      const newPending = new Map(state.pending);
      newPending.set(id, candidate);
      return {
        ...state,
        pending: newPending,
      };
    }

    case 'REMOVE':
    case 'DESELECT': {
      const id = action.candidateId || action.id || action.candidate?.candidate_id || action.candidate?.frame_id;
      if (!id || !state.pending.has(id)) {
        return state;
      }
      const newPending = new Map(state.pending);
      newPending.delete(id);
      return {
        ...state,
        pending: newPending,
      };
    }

    case 'TOGGLE': {
      const candidate = action.candidate;
      if (!candidate || typeof candidate !== 'object') return state;
      const id = candidate.candidate_id || candidate.frame_id;
      if (!id) return state;
      if (state.pending.has(id)) {
        const newPending = new Map(state.pending);
        newPending.delete(id);
        return {
          ...state,
          pending: newPending,
        };
      }
      if (state.submitted.has(id)) {
        return state;
      }
      if (!isValidCandidate(candidate)) {
        return state;
      }
      const newPending = new Map(state.pending);
      newPending.set(id, candidate);
      return {
        ...state,
        pending: newPending,
      };
    }

    case 'CLEAR_PENDING': {
      if (state.pending.size === 0 && state.unknownBatch === null) {
        return state;
      }
      return {
        ...state,
        pending: new Map(),
        unknownBatch: null,
      };
    }

    case 'RECORDED': {
      const ids = Array.isArray(action.candidateIds) ? action.candidateIds : [];
      if (ids.length === 0) {
        return {
          ...state,
          unknownBatch: null,
        };
      }
      const newPending = new Map(state.pending);
      const newSubmitted = new Map(state.submitted);
      for (const id of ids) {
        if (newPending.has(id)) {
          newSubmitted.set(id, newPending.get(id));
          newPending.delete(id);
        } else {
          newSubmitted.set(id, true);
        }
      }
      return {
        ...state,
        pending: newPending,
        submitted: newSubmitted,
        unknownBatch: null,
      };
    }

    case 'UNKNOWN': {
      const ids = Array.isArray(action.candidateIds) ? [...action.candidateIds] : [];
      return {
        ...state,
        unknownBatch: ids,
      };
    }

    case 'MARK_UNKNOWN_RECORDED': {
      const ids = state.unknownBatch;
      if (!Array.isArray(ids) || ids.length === 0) {
        return state;
      }
      const newPending = new Map(state.pending);
      const newSubmitted = new Map(state.submitted);
      for (const id of ids) {
        if (newPending.has(id)) {
          newSubmitted.set(id, newPending.get(id));
          newPending.delete(id);
        } else {
          newSubmitted.set(id, true);
        }
      }
      return {
        ...state,
        pending: newPending,
        submitted: newSubmitted,
        unknownBatch: null,
      };
    }

    case 'RESET_FOR_SCOPE': {
      return createInitialAvsSelectionState(action.taskScopeKey ?? null);
    }

    default:
      return state;
  }
};
