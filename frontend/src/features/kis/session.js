/**
 * Pure session state model for stateless multimodal KIS operations.
 *
 * This module manages immutable transitions for base_intent + operation progression,
 * eliminating server-side session state and raw clue lists.
 */

import {
  extractEventPatches,
  extractGlobalRewriteInstruction,
} from './parser';

export const createInitialKisSessionState = () => ({
  draft: '',
  currentIntent: null,
  revision: 0,
  stagedImages: {},
  pendingOperation: null,
  isSearching: false,
  error: null,
  mode: 'live',
});

export const setDraft = (state, draft) => ({
  ...state,
  draft,
  error: null,
});

export const setMode = (state, mode) => ({
  ...state,
  mode,
});

export const stageImage = (state, eventId, imageRef) => {
  const current = state.stagedImages[eventId] || [];
  return {
    ...state,
    stagedImages: {
      ...state.stagedImages,
      [eventId]: [...current, imageRef],
    },
  };
};

export const unstageImage = (state, eventId, assetId) => {
  const current = state.stagedImages[eventId] || [];
  const updated = current.filter(
    (img) => (img.asset_id || img.id) !== assetId,
  );
  const nextStaged = { ...state.stagedImages };
  if (updated.length > 0) {
    nextStaged[eventId] = updated;
  } else {
    delete nextStaged[eventId];
  }
  return {
    ...state,
    stagedImages: nextStaged,
  };
};

export const clearStagedImages = (state) => ({
  ...state,
  stagedImages: {},
});

export const prepareSemanticRequest = (state, preview) => {
  if (state.mode === 'replay') {
    throw new Error('Cannot search in replay mode');
  }
  if (!preview || preview.kind === 'invalid' || preview.error) {
    throw new Error(preview?.error || 'Invalid command');
  }

  let operation;
  if (preview.kind === 'initial_resolve') {
    if (preview.affectedEventIds.length > 0) {
      const patches = extractEventPatches(state.draft).map((p) => ({
        event_id: p.event_id,
        instruction: p.instruction,
        add_image_ids: (state.stagedImages[p.event_id] || []).map(
          (img) => img.asset_id || img.id || img,
        ),
        remove_image_ids: [],
      }));
      operation = {
        kind: 'initial_resolve',
        patches,
      };
    } else {
      const allImageRefs = Object.values(state.stagedImages).flat();
      operation = {
        kind: 'initial_resolve',
        text: (state.draft || '').trim(),
        image_refs: allImageRefs,
        patches: [],
      };
    }
  } else if (preview.kind === 'patch_events') {
    const patches = extractEventPatches(state.draft).map((p) => ({
      event_id: p.event_id,
      instruction: p.instruction,
      add_image_ids: (state.stagedImages[p.event_id] || []).map(
        (img) => img.asset_id || img.id || img,
      ),
      remove_image_ids: [],
    }));
    operation = {
      kind: 'patch_events',
      patches,
    };
  } else if (preview.kind === 'global_rewrite') {
    const instruction = extractGlobalRewriteInstruction(state.draft);
    operation = {
      kind: 'global_rewrite',
      instruction,
    };
  } else {
    throw new Error(`Unsupported operation preview kind: ${preview.kind}`);
  }

  const nextState = {
    ...state,
    isSearching: true,
    error: null,
    pendingOperation: operation,
  };

  const requestPayload = {
    baseIntent: state.currentIntent,
    expectedRevision: state.revision,
    operation,
  };

  return { nextState, requestPayload };
};

export const prepareSearchOnlyRequest = (state) => {
  if (state.mode === 'replay') {
    throw new Error('Cannot search in replay mode');
  }
  if (!state.currentIntent) {
    throw new Error('Cannot run search_only without active intent');
  }

  const operation = { kind: 'search_only' };
  const nextState = {
    ...state,
    isSearching: true,
    error: null,
    pendingOperation: operation,
  };
  const requestPayload = {
    baseIntent: state.currentIntent,
    expectedRevision: state.revision,
    operation,
  };

  return { nextState, requestPayload };
};

export const commitSearchSuccess = (state, response) => {
  const intent = response?.intent || state.currentIntent;
  const revision = typeof intent?.revision === 'number' ? intent.revision : state.revision;

  return {
    ...state,
    draft: '',
    stagedImages: {},
    revision,
    currentIntent: intent,
    pendingOperation: null,
    isSearching: false,
    error: null,
  };
};

export const commitSearchFailure = (state, error) => ({
  ...state,
  pendingOperation: null,
  isSearching: false,
  error: error?.message || (typeof error === 'string' ? error : 'Search failed'),
});

export const resetKisSession = () => createInitialKisSessionState();
