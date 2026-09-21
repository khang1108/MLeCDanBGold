/**
 * Pure session state and reducers for transactional KIS chat feedback.
 *
 * Manages conversational turns, context chips, pending requests, and Undo history.
 */

export const createInitialFeedbackSession = () => ({
  sessionId: null,
  feedbackRevision: 0,
  status: 'idle', // 'idle' | 'pending' | 'applied' | 'clarification' | 'exhausted' | 'error'
  messages: [],
  currentIntent: null,
  retrievalOverrides: {},
  results: [],
  evidenceSnapshotId: null,
  trail: null,
  selectedContext: null, // { resultId, eventId, frameId } | null
  canUndo: false,
  pendingRequestId: null,
  error: null,
});

export const initFeedbackSession = (state, { sessionId, feedbackRevision, state: serverState, originalQuery }) => {
  let initialMessages = (Array.isArray(state?.messages) && state.messages.length > 0)
    ? [...state.messages]
    : [];
  if (initialMessages.length === 0) {
    if (originalQuery) {
      initialMessages.push({
        id: 'msg_initial_query',
        role: 'user',
        text: originalQuery,
      });
    }
    initialMessages.push({
      id: 'msg_session_opened',
      role: 'assistant',
      text: serverState?.assistant_message || 'Feedback session opened. How can I help refine the results?',
    });
  }

  return {
    ...state,
    sessionId: sessionId || serverState?.session_id || state.sessionId,
    feedbackRevision: feedbackRevision ?? serverState?.feedback_revision ?? 1,
    status: 'applied',
    messages: initialMessages,
    currentIntent: serverState?.intent || state.currentIntent,
    retrievalOverrides: serverState?.retrieval_overrides || {},
    results: serverState?.results || [],
    evidenceSnapshotId: serverState?.evidence_snapshot_id || null,
    trail: serverState?.trail || null,
    canUndo: !!serverState?.can_undo,
    pendingRequestId: null,
    error: null,
  };
};

export const startFeedbackTurn = (state, { requestId, message, selectedContext }) => {
  const userMsg = {
    id: requestId,
    role: 'user',
    text: message,
    context: selectedContext !== undefined ? selectedContext : state.selectedContext,
  };

  return {
    ...state,
    status: 'pending',
    pendingRequestId: requestId,
    error: null,
    messages: [...state.messages, userMsg],
  };
};

export const applyFeedbackSuccess = (state, response, { requestId } = {}) => {
  // Stale request guard
  if (requestId && state.pendingRequestId && requestId !== state.pendingRequestId) {
    return state;
  }
  // Stale revision guard
  if (typeof response.feedback_revision === 'number' && response.feedback_revision < state.feedbackRevision) {
    return state;
  }

  const assistantMsg = {
    id: `asst_${Date.now()}`,
    role: 'assistant',
    text: response.assistant_message || (response.status === 'clarification' ? 'Please clarify your feedback.' : 'Applied feedback.'),
    status: response.status,
    scope: response.scope,
    changedEventIds: response.changed_event_ids || [],
  };

  if (response.status === 'clarification') {
    return {
      ...state,
      status: 'clarification',
      pendingRequestId: null,
      error: null,
      messages: [...state.messages, assistantMsg],
    };
  }

  const isGlobalChange = response.scope === 'all_videos';

  return {
    ...state,
    status: response.status || 'applied',
    feedbackRevision: response.feedback_revision ?? state.feedbackRevision + 1,
    currentIntent: response.status === 'proposal' ? state.currentIntent : (response.intent || state.currentIntent),
    retrievalOverrides: response.retrieval_overrides !== undefined ? response.retrieval_overrides : state.retrievalOverrides,
    results: response.results !== undefined ? response.results : state.results,
    evidenceSnapshotId: response.evidence_snapshot_id || state.evidenceSnapshotId,
    trail: response.trail !== undefined ? response.trail : state.trail,
    canUndo: !!response.can_undo,
    selectedContext: isGlobalChange ? null : state.selectedContext,
    pendingRequestId: null,
    error: null,
    messages: [...state.messages, assistantMsg],
  };
};

export const applyFeedbackFailure = (state, error, { requestId } = {}) => {
  if (requestId && state.pendingRequestId && requestId !== state.pendingRequestId) {
    return state;
  }

  return {
    ...state,
    status: 'error',
    error: error?.message || String(error || 'Feedback turn failed'),
    pendingRequestId: null,
  };
};

export const applyUndoSuccess = (state, response) => {
  const undoMsg = {
    id: `undo_${Date.now()}`,
    role: 'assistant',
    text: response.assistant_message || 'Reverted previous feedback turn.',
  };

  return {
    ...state,
    status: 'applied',
    feedbackRevision: response.feedback_revision,
    currentIntent: response.intent || state.currentIntent,
    retrievalOverrides: response.retrieval_overrides || {},
    results: response.results || [],
    evidenceSnapshotId: response.evidence_snapshot_id || state.evidenceSnapshotId,
    trail: response.trail || null,
    canUndo: !!response.can_undo,
    pendingRequestId: null,
    error: null,
    messages: [...state.messages, undoMsg],
  };
};

export const setSelectedContext = (state, context) => ({
  ...state,
  selectedContext: context ? { ...context } : null,
});

export const clearSelectedContext = (state) => ({
  ...state,
  selectedContext: null,
});

export const resetFeedbackSession = () => createInitialFeedbackSession();
