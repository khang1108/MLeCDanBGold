import {
  avsSelectionReducer,
  candidateToTemporalAnswer,
  createInitialAvsSelectionState,
} from './selectionState';

const candidate = (id, videoId = 'V1', timestampMs = 1000) => ({
  candidate_id: id,
  frame_id: id,
  video_id: videoId,
  timestamp_ms: timestampMs,
});

describe('avsSelectionReducer', () => {
  test('reselecting the same canonical frame does not duplicate pending state', () => {
    let state = createInitialAvsSelectionState();
    state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    expect(state.pending.size).toBe(1);
  });

  test('cannot select candidate that has already been submitted', () => {
    let state = createInitialAvsSelectionState();
    state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    state = avsSelectionReducer(state, { type: 'RECORDED', candidateIds: ['f1'] });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    expect(state.pending.size).toBe(0);
    expect(state.submitted.has('f1')).toBe(true);
  });

  test('deselect removes candidate from pending', () => {
    let state = createInitialAvsSelectionState();
    state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    state = avsSelectionReducer(state, { type: 'DESELECT', candidateId: 'f1' });
    expect(state.pending.size).toBe(0);
  });

  test('toggle selects unselected candidate and deselects selected candidate', () => {
    let state = createInitialAvsSelectionState();
    state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
    state = avsSelectionReducer(state, { type: 'TOGGLE', candidate: candidate('f1') });
    expect(state.pending.size).toBe(1);
    expect(state.pending.has('f1')).toBe(true);

    state = avsSelectionReducer(state, { type: 'TOGGLE', candidate: candidate('f1') });
    expect(state.pending.size).toBe(0);
  });

  test('remove alias removes candidate from pending', () => {
    let state = createInitialAvsSelectionState();
    state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    state = avsSelectionReducer(state, { type: 'REMOVE', candidateId: 'f1' });
    expect(state.pending.size).toBe(0);
  });

  test('clear pending clears pending and unknownBatch', () => {
    let state = createInitialAvsSelectionState();
    state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    state = avsSelectionReducer(state, { type: 'UNKNOWN', candidateIds: ['f1'] });
    state = avsSelectionReducer(state, { type: 'CLEAR_PENDING' });
    expect(state.pending.size).toBe(0);
    expect(state.unknownBatch).toBeNull();
  });

  test('recorded batch clears only pending items and marks them submitted', () => {
    let state = createInitialAvsSelectionState();
    state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    state = avsSelectionReducer(state, { type: 'RECORDED', candidateIds: ['f1'] });
    expect(state.pending.size).toBe(0);
    expect(state.submitted.has('f1')).toBe(true);
  });

  test('unknown batch keeps pending and stores immutable retry snapshot', () => {
    let state = createInitialAvsSelectionState();
    state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    state = avsSelectionReducer(state, { type: 'UNKNOWN', candidateIds: ['f1'] });
    expect(state.pending.has('f1')).toBe(true);
    expect(state.unknownBatch).toEqual(['f1']);
  });

  test('mark unknown recorded moves unknownBatch IDs to submitted', () => {
    let state = createInitialAvsSelectionState();
    state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    state = avsSelectionReducer(state, { type: 'UNKNOWN', candidateIds: ['f1'] });
    state = avsSelectionReducer(state, { type: 'MARK_UNKNOWN_RECORDED' });
    expect(state.pending.size).toBe(0);
    expect(state.submitted.has('f1')).toBe(true);
    expect(state.unknownBatch).toBeNull();
  });

  test('reset for scope produces new empty state bound to new scope', () => {
    let state = createInitialAvsSelectionState();
    state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
    state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
    state = avsSelectionReducer(state, { type: 'RESET_FOR_SCOPE', taskScopeKey: 'scope-2' });
    expect(state.taskScopeKey).toBe('scope-2');
    expect(state.pending.size).toBe(0);
    expect(state.submitted.size).toBe(0);
    expect(state.unknownBatch).toBeNull();
  });

  test('candidateToTemporalAnswer formats canonical candidate', () => {
    const answer = candidateToTemporalAnswer(candidate('f1', 'V01', 5000));
    expect(answer).toEqual({
      kind: 'TEMPORAL',
      video_id: 'V01',
      start_ms: 5000,
      end_ms: 5000,
    });
  });
});
