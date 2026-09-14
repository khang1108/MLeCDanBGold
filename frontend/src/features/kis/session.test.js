import {
  createInitialKisSessionState,
  setDraft,
  prepareSearchRequest,
  commitSearchSuccess,
  commitSearchFailure,
  resetKisSession,
} from './session';

describe('KIS revisioned session state transitions', () => {
  test('initial state has empty draft, zero revision, empty inputs, and null intent', () => {
    const state = createInitialKisSessionState();
    expect(state).toEqual({
      draft: '',
      committedInputs: [],
      revision: 0,
      currentIntent: null,
      isSearching: false,
      error: null,
    });
  });

  test('submit Q1: prepares request revision 0, then commits revision 1', () => {
    let state = createInitialKisSessionState();
    state = setDraft(state, '  A woman stands in a kitchen.  ');
    expect(state.draft).toBe('  A woman stands in a kitchen.  ');

    const { nextState, requestPayload } = prepareSearchRequest(state);
    expect(nextState.isSearching).toBe(true);
    expect(nextState.error).toBe(null);
    expect(requestPayload).toEqual({
      inputs: [{ text: 'A woman stands in a kitchen.' }],
      expectedRevision: 0,
    });

    const mockIntent1 = {
      revision: 1,
      inputs: ['A woman stands in a kitchen.'],
      language: 'en',
      query_text: 'A woman stands in a kitchen.',
      entities: [{ id: 'X1', kind: 'person', description: 'woman' }],
      events: [{ id: 'E1', text: 'A woman stands in a kitchen.', bindings: [] }],
      temporal_edges: [],
    };

    state = commitSearchSuccess(nextState, { intent: mockIntent1 });
    expect(state).toEqual({
      draft: '',
      committedInputs: ['A woman stands in a kitchen.'],
      revision: 1,
      currentIntent: mockIntent1,
      isSearching: false,
      error: null,
    });
  });

  test('revision 1: submit Q2 prepares request with [Q1, Q2] and expectedRevision 1, then commits revision 2', () => {
    const mockIntent1 = {
      revision: 1,
      inputs: ['A woman stands in a kitchen.'],
      language: 'en',
      query_text: 'A woman stands in a kitchen.',
      entities: [{ id: 'X1', kind: 'person', description: 'woman' }],
      events: [{ id: 'E1', text: 'A woman stands in a kitchen.', bindings: [] }],
      temporal_edges: [],
    };

    let state = {
      draft: '',
      committedInputs: ['A woman stands in a kitchen.'],
      revision: 1,
      currentIntent: mockIntent1,
      isSearching: false,
      error: null,
    };

    state = setDraft(state, 'She speaks to a chef.');
    const { nextState, requestPayload } = prepareSearchRequest(state);
    expect(nextState.isSearching).toBe(true);
    expect(requestPayload).toEqual({
      inputs: [
        { text: 'A woman stands in a kitchen.' },
        { text: 'She speaks to a chef.' },
      ],
      expectedRevision: 1,
    });

    const mockIntent2 = {
      revision: 2,
      inputs: ['A woman stands in a kitchen.', 'She speaks to a chef.'],
      language: 'en',
      query_text: 'A woman stands in a kitchen and speaks to a chef.',
      entities: [
        { id: 'X1', kind: 'person', description: 'woman' },
        { id: 'X2', kind: 'person', description: 'chef' },
      ],
      events: [
        { id: 'E1', text: 'A woman stands in a kitchen.', bindings: [] },
        { id: 'E2', text: 'She speaks to a chef.', bindings: [] },
      ],
      temporal_edges: [{ source: 'E1', relation: 'before', target: 'E2' }],
    };

    state = commitSearchSuccess(nextState, { intent: mockIntent2 });
    expect(state).toEqual({
      draft: '',
      committedInputs: ['A woman stands in a kitchen.', 'She speaks to a chef.'],
      revision: 2,
      currentIntent: mockIntent2,
      isSearching: false,
      error: null,
    });
  });

  test('request failure: keeps previous revision and committed inputs, retains draft for editing', () => {
    const mockIntent1 = {
      revision: 1,
      inputs: ['A woman stands in a kitchen.'],
      language: 'en',
      query_text: 'A woman stands in a kitchen.',
      entities: [{ id: 'X1', kind: 'person', description: 'woman' }],
      events: [{ id: 'E1', text: 'A woman stands in a kitchen.', bindings: [] }],
      temporal_edges: [],
    };

    const previousState = {
      draft: 'She speaks to a chef.',
      committedInputs: ['A woman stands in a kitchen.'],
      revision: 1,
      currentIntent: mockIntent1,
      isSearching: true,
      error: null,
    };

    const failedState = commitSearchFailure(previousState, new Error('Network error (502)'));
    expect(failedState).toEqual({
      draft: 'She speaks to a chef.',
      committedInputs: ['A woman stands in a kitchen.'],
      revision: 1,
      currentIntent: mockIntent1,
      isSearching: false,
      error: 'Network error (502)',
    });
  });

  test('reset: clears inputs, revision to 0, intent to null, and draft to empty', () => {
    const mockIntent1 = {
      revision: 1,
      inputs: ['A woman stands in a kitchen.'],
      language: 'en',
      query_text: 'A woman stands in a kitchen.',
      entities: [{ id: 'X1', kind: 'person', description: 'woman' }],
      events: [{ id: 'E1', text: 'A woman stands in a kitchen.', bindings: [] }],
      temporal_edges: [],
    };

    const currentState = {
      draft: 'Pending clue...',
      committedInputs: ['A woman stands in a kitchen.'],
      revision: 1,
      currentIntent: mockIntent1,
      isSearching: false,
      error: 'Some error',
    };

    const resetState = resetKisSession(currentState);
    expect(resetState).toEqual({
      draft: '',
      committedInputs: [],
      revision: 0,
      currentIntent: null,
      isSearching: false,
      error: null,
    });
  });

  test('prepareSearchRequest throws if draft is empty or blank', () => {
    const state = createInitialKisSessionState();
    expect(() => prepareSearchRequest(state)).toThrow('Search clue cannot be empty');
    expect(() => prepareSearchRequest(setDraft(state, '   '))).toThrow('Search clue cannot be empty');
  });
});
