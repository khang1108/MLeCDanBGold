/**
 * Pure session state model for revisioned multi-clue KIS search.
 *
 * This module manages immutable transitions across participant clue revisions,
 * ensuring failed requests preserve previous committed inputs and draft text.
 */

export const createInitialKisSessionState = () => ({
  draft: '',
  committedInputs: [],
  revision: 0,
  currentIntent: null,
  isSearching: false,
  error: null,
});

export const setDraft = (state, draft) => ({
  ...state,
  draft,
  error: null,
});

export const prepareSearchRequest = (state) => {
  const trimmedDraft = (state.draft || '').trim();
  if (!trimmedDraft) {
    throw new Error('Search clue cannot be empty');
  }

  const allTexts = [...state.committedInputs, trimmedDraft];
  const requestPayload = {
    inputs: allTexts.map((text) => ({ text })),
    expectedRevision: state.revision,
  };

  const nextState = {
    ...state,
    isSearching: true,
    error: null,
  };

  return { nextState, requestPayload };
};

export const commitSearchSuccess = (state, response) => {
  const intent = response?.intent;
  const committedInputs = Array.isArray(intent?.inputs)
    ? intent.inputs.slice()
    : state.committedInputs;

  return {
    ...state,
    draft: '',
    committedInputs,
    revision: typeof intent?.revision === 'number' ? intent.revision : state.committedInputs.length,
    currentIntent: intent || null,
    isSearching: false,
    error: null,
  };
};

export const commitSearchFailure = (state, error) => ({
  ...state,
  isSearching: false,
  error: error?.message || (typeof error === 'string' ? error : 'Search failed'),
});

export const resetKisSession = () => createInitialKisSessionState();
