export const createInitialQueryHypothesisState = () => ({
  sessionId: null,
  intent: null,
  queryRevision: 0,
  preview: null,
  resultsQueryRevision: null,
  isMutating: false,
  error: null,
  canUndo: false,
});

export const receiveOpenedHypothesis = (state, payload) => ({
  ...state,
  sessionId: payload.session_id ?? state.sessionId,
  queryRevision: payload.query_revision ?? payload.intent?.revision ?? 0,
  intent: payload.intent ?? state.intent,
  canUndo: Boolean(payload.can_undo),
  preview: null,
  error: null,
});

export const receivePreview = (state, previewPayload) => ({
  ...state,
  preview: previewPayload,
  error: null,
});

export const clearPreview = (state) => ({
  ...state,
  preview: null,
});

export const receiveCommit = (state, commitPayload) => ({
  ...state,
  sessionId: commitPayload.session_id ?? state.sessionId,
  queryRevision: commitPayload.query_revision ?? commitPayload.intent?.revision ?? state.queryRevision,
  intent: commitPayload.intent ?? state.intent,
  canUndo: Boolean(commitPayload.can_undo),
  preview: null,
  error: null,
});

export const markSearchResults = (state, revision) => ({
  ...state,
  resultsQueryRevision: revision !== undefined ? revision : (state.intent?.revision ?? state.queryRevision),
});

export const isResultsStale = (state) => {
  if (state.resultsQueryRevision === null || state.resultsQueryRevision === undefined) {
    return false;
  }
  const currentRevision = state.intent?.revision ?? state.queryRevision;
  return state.resultsQueryRevision !== currentRevision;
};

export const setError = (state, error) => ({
  ...state,
  error,
});

export const setMutating = (state, isMutating) => ({
  ...state,
  isMutating,
});
