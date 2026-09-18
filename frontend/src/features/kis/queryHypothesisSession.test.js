import {
  createInitialQueryHypothesisState,
  receiveOpenedHypothesis,
  receivePreview,
  clearPreview,
  receiveCommit,
  markSearchResults,
  isResultsStale,
} from './queryHypothesisSession';

describe('queryHypothesisSession client state helpers', () => {
  test('preview does not change canonical revision or intent', () => {
    let state = receiveOpenedHypothesis(createInitialQueryHypothesisState(), {
      session_id: 'qh_1',
      query_revision: 1,
      intent: { revision: 1, events: [] },
      can_undo: false,
    });
    state = receivePreview(state, {
      base_revision: 1,
      intent: { revision: 2, events: [{ id: 'E1', text: 'previewed' }] },
    });
    expect(state.intent.revision).toBe(1);
    expect(state.preview.intent.revision).toBe(2);
    expect(state.preview.intent.events[0].text).toBe('previewed');
  });

  test('clearPreview drops local preview', () => {
    let state = receiveOpenedHypothesis(createInitialQueryHypothesisState(), {
      session_id: 'qh_1',
      query_revision: 1,
      intent: { revision: 1, events: [] },
    });
    state = receivePreview(state, {
      base_revision: 1,
      intent: { revision: 2, events: [{ id: 'E1' }] },
    });
    expect(state.preview).not.toBeNull();
    state = clearPreview(state);
    expect(state.preview).toBeNull();
  });

  test('committed edit makes existing results stale without deleting them', () => {
    let state = {
      ...createInitialQueryHypothesisState(),
      intent: { revision: 2 },
      resultsQueryRevision: 2,
    };
    state = receiveCommit(state, {
      session_id: 'qh_1',
      query_revision: 3,
      intent: { revision: 3 },
      can_undo: true,
    });
    expect(state.resultsQueryRevision).toBe(2);
    expect(state.intent.revision).toBe(3);
    expect(state.preview).toBeNull();
    expect(isResultsStale(state)).toBe(true);
  });

  test('markSearchResults updates resultsQueryRevision to match intent revision', () => {
    let state = {
      ...createInitialQueryHypothesisState(),
      intent: { revision: 3 },
      resultsQueryRevision: 2,
    };
    expect(isResultsStale(state)).toBe(true);
    state = markSearchResults(state, 3);
    expect(state.resultsQueryRevision).toBe(3);
    expect(isResultsStale(state)).toBe(false);
  });
});
