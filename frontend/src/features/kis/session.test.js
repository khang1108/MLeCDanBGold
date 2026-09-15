import {
  createInitialKisSessionState,
  setDraft,
  setMode,
  stageImage,
  unstageImage,
  clearStagedImages,
  prepareSemanticRequest,
  prepareSearchOnlyRequest,
  commitSearchSuccess,
  commitSearchFailure,
  resetKisSession,
} from './session';

describe('KIS session state model', () => {
  const MOCK_INTENT_REV1 = {
    revision: 1,
    language: 'en',
    query_text: 'chef cooking pasta',
    entities: [{ id: 'X1', kind: 'person', description: 'chef' }],
    events: [{ id: 'E1', text: 'chef cooking pasta', images: [], bindings: [] }],
    temporal_edges: [],
  };

  const MOCK_INTENT_REV2 = {
    revision: 2,
    language: 'en',
    query_text: 'chef cooking pasta then plating it',
    entities: [
      { id: 'X1', kind: 'person', description: 'chef' },
      { id: 'X2', kind: 'food', description: 'pasta' },
    ],
    events: [
      { id: 'E1', text: 'chef cooking pasta', images: [], bindings: [] },
      { id: 'E2', text: 'chef plating pasta', images: [], bindings: [] },
    ],
    temporal_edges: [{ source: 'E1', target: 'E2', relation: 'before' }],
  };

  test('initial state matches expected target shape', () => {
    const state = createInitialKisSessionState();
    expect(state).toEqual({
      draft: '',
      currentIntent: null,
      revision: 0,
      stagedImages: {},
      pendingOperation: null,
      isSearching: false,
      error: null,
      mode: 'live',
    });
  });

  test('setDraft updates draft and clears error', () => {
    let state = { ...createInitialKisSessionState(), error: 'Previous error' };
    state = setDraft(state, 'new draft text');
    expect(state.draft).toBe('new draft text');
    expect(state.error).toBe(null);
  });

  test('staging and unstaging images', () => {
    let state = createInitialKisSessionState();
    const image1 = { asset_id: 'img_1', file_name: 'ref1.jpg' };
    const image2 = { asset_id: 'img_2', file_name: 'ref2.jpg' };

    state = stageImage(state, 'E1', image1);
    state = stageImage(state, 'E1', image2);
    expect(state.stagedImages.E1).toEqual([image1, image2]);

    state = unstageImage(state, 'E1', 'img_1');
    expect(state.stagedImages.E1).toEqual([image2]);

    state = clearStagedImages(state);
    expect(state.stagedImages).toEqual({});
  });

  test('prepareSemanticRequest for initial natural text resolve', () => {
    let state = createInitialKisSessionState();
    state = setDraft(state, 'chef cooking pasta');

    const preview = {
      kind: 'initial_resolve',
      affectedEventIds: [],
      error: null,
    };

    const { nextState, requestPayload } = prepareSemanticRequest(state, preview);

    expect(nextState.isSearching).toBe(true);
    expect(nextState.error).toBe(null);
    expect(nextState.pendingOperation).toEqual({
      kind: 'initial_resolve',
      text: 'chef cooking pasta',
      image_refs: [],
      patches: [],
    });
    expect(requestPayload).toEqual({
      baseIntent: null,
      expectedRevision: 0,
      operation: {
        kind: 'initial_resolve',
        text: 'chef cooking pasta',
        image_refs: [],
        patches: [],
      },
    });
  });

  test('prepareSemanticRequest for patch_events with staged images', () => {
    let state = {
      ...createInitialKisSessionState(),
      currentIntent: MOCK_INTENT_REV1,
      revision: 1,
      draft: 'E1: chef stirs sauce\nE2: chef plates dish',
      stagedImages: {
        E2: [{ asset_id: 'asset_e2_1' }],
      },
    };

    const preview = {
      kind: 'patch_events',
      affectedEventIds: ['E1', 'E2'],
      error: null,
    };

    const { nextState, requestPayload } = prepareSemanticRequest(state, preview);

    expect(nextState.isSearching).toBe(true);
    expect(requestPayload.baseIntent).toEqual(MOCK_INTENT_REV1);
    expect(requestPayload.expectedRevision).toBe(1);
    expect(requestPayload.operation).toEqual({
      kind: 'patch_events',
      patches: [
        {
          event_id: 'E1',
          instruction: 'chef stirs sauce',
          add_image_ids: [],
          remove_image_ids: [],
        },
        {
          event_id: 'E2',
          instruction: 'chef plates dish',
          add_image_ids: ['asset_e2_1'],
          remove_image_ids: [],
        },
      ],
    });
  });

  test('prepareSemanticRequest for global_rewrite', () => {
    let state = {
      ...createInitialKisSessionState(),
      currentIntent: MOCK_INTENT_REV1,
      revision: 1,
      draft: '/llm-rewrite\nchange scenario to outdoor BBQ',
    };

    const preview = {
      kind: 'global_rewrite',
      affectedEventIds: ['E1'],
      error: null,
    };

    const { nextState, requestPayload } = prepareSemanticRequest(state, preview);

    expect(requestPayload.operation).toEqual({
      kind: 'global_rewrite',
      instruction: 'change scenario to outdoor BBQ',
    });
  });

  test('prepareSemanticRequest throws if preview has error or is invalid', () => {
    const state = createInitialKisSessionState();
    expect(() => prepareSemanticRequest(state, { kind: 'invalid', error: 'Missing E4' })).toThrow(/Missing E4/);
    expect(() => prepareSemanticRequest(state, null)).toThrow(/invalid/i);
  });

  test('replay mode cannot create live search payload', () => {
    let state = { ...createInitialKisSessionState(), mode: 'replay', draft: 'test' };
    expect(() => prepareSemanticRequest(state, { kind: 'initial_resolve', affectedEventIds: [], error: null })).toThrow(/replay/i);
    expect(() => prepareSearchOnlyRequest(state)).toThrow(/replay/i);
  });

  test('prepareSearchOnlyRequest keeps current intent and revision', () => {
    const state = {
      ...createInitialKisSessionState(),
      currentIntent: MOCK_INTENT_REV1,
      revision: 1,
    };

    const { nextState, requestPayload } = prepareSearchOnlyRequest(state);

    expect(nextState.isSearching).toBe(true);
    expect(nextState.pendingOperation).toEqual({ kind: 'search_only' });
    expect(requestPayload).toEqual({
      baseIntent: MOCK_INTENT_REV1,
      expectedRevision: 1,
      operation: { kind: 'search_only' },
    });
  });

  test('prepareSearchOnlyRequest throws if no active intent', () => {
    const state = createInitialKisSessionState();
    expect(() => prepareSearchOnlyRequest(state)).toThrow(/active intent/i);
  });

  test('commitSearchSuccess increments revision and clears draft/stagedImages', () => {
    const searchingState = {
      ...createInitialKisSessionState(),
      draft: 'E2: chef plates dish',
      stagedImages: { E2: [{ asset_id: 'a1' }] },
      currentIntent: MOCK_INTENT_REV1,
      revision: 1,
      pendingOperation: { kind: 'patch_events' },
      isSearching: true,
    };

    const response = {
      intent: MOCK_INTENT_REV2,
      operation_summary: { kind: 'patch_events', affected_event_ids: ['E2'] },
      results: [],
      latency: { total_ms: 25 },
    };

    const nextState = commitSearchSuccess(searchingState, response);

    expect(nextState.draft).toBe('');
    expect(nextState.stagedImages).toEqual({});
    expect(nextState.revision).toBe(2);
    expect(nextState.currentIntent).toEqual(MOCK_INTENT_REV2);
    expect(nextState.pendingOperation).toBe(null);
    expect(nextState.isSearching).toBe(false);
    expect(nextState.error).toBe(null);
  });

  test('commitSearchSuccess preserves revision on search_only', () => {
    const searchingState = {
      ...createInitialKisSessionState(),
      currentIntent: MOCK_INTENT_REV1,
      revision: 1,
      pendingOperation: { kind: 'search_only' },
      isSearching: true,
    };

    const response = {
      intent: MOCK_INTENT_REV1,
      operation_summary: { kind: 'search_only', affected_event_ids: [] },
      results: [],
      latency: { total_ms: 10 },
    };

    const nextState = commitSearchSuccess(searchingState, response);
    expect(nextState.revision).toBe(1);
    expect(nextState.currentIntent).toEqual(MOCK_INTENT_REV1);
  });

  test('commitSearchFailure preserves draft, staged images, and current intent', () => {
    const searchingState = {
      ...createInitialKisSessionState(),
      draft: 'E2: chef plates dish',
      stagedImages: { E2: [{ asset_id: 'a1' }] },
      currentIntent: MOCK_INTENT_REV1,
      revision: 1,
      pendingOperation: { kind: 'patch_events' },
      isSearching: true,
    };

    const failedState = commitSearchFailure(searchingState, new Error('Network error (502)'));

    expect(failedState.draft).toBe('E2: chef plates dish');
    expect(failedState.stagedImages).toEqual({ E2: [{ asset_id: 'a1' }] });
    expect(failedState.currentIntent).toEqual(MOCK_INTENT_REV1);
    expect(failedState.revision).toBe(1);
    expect(failedState.pendingOperation).toBe(null);
    expect(failedState.isSearching).toBe(false);
    expect(failedState.error).toBe('Network error (502)');
  });

  test('resetKisSession resets to initial state', () => {
    const state = {
      draft: 'draft',
      currentIntent: MOCK_INTENT_REV1,
      revision: 1,
      stagedImages: { E1: [{ asset_id: 'a1' }] },
      pendingOperation: null,
      isSearching: false,
      error: 'error',
      mode: 'live',
    };

    expect(resetKisSession()).toEqual(createInitialKisSessionState());
  });
});
