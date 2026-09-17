import {
  createInitialFeedbackSession,
  initFeedbackSession,
  startFeedbackTurn,
  applyFeedbackSuccess,
  applyFeedbackFailure,
  applyUndoSuccess,
  setSelectedContext,
  clearSelectedContext,
} from './feedbackSession';

describe('feedbackSession state reducers', () => {
  const SAMPLE_INTENT = {
    revision: 1,
    query_text: 'A woman in red dress walks into a room.',
    events: [
      { id: 'E1', text: 'A woman in red dress walks into a room.', images: [] },
    ],
    temporal_edges: [],
  };

  const INITIAL_RESULTS = [
    { result_id: 'r1', video_id: 'v1', frame_idx: 10, timestamp_ms: 1000, score: 0.9 },
  ];

  test('initializes feedback session correctly from server state', () => {
    const session = initFeedbackSession(createInitialFeedbackSession(), {
      sessionId: 'fbs_123',
      feedbackRevision: 1,
      state: {
        session_id: 'fbs_123',
        status: 'applied',
        feedback_revision: 1,
        intent: SAMPLE_INTENT,
        results: INITIAL_RESULTS,
        evidence_snapshot_id: 'snap_1',
        can_undo: false,
      },
    });

    expect(session.sessionId).toBe('fbs_123');
    expect(session.feedbackRevision).toBe(1);
    expect(session.currentIntent).toEqual(SAMPLE_INTENT);
    expect(session.results).toEqual(INITIAL_RESULTS);
    expect(session.canUndo).toBe(false);
  });

  test('startFeedbackTurn adds user message and retains selected context', () => {
    const base = initFeedbackSession(createInitialFeedbackSession(), {
      sessionId: 'fbs_123',
      feedbackRevision: 1,
      state: {
        session_id: 'fbs_123',
        status: 'applied',
        feedback_revision: 1,
        intent: SAMPLE_INTENT,
        results: INITIAL_RESULTS,
        evidence_snapshot_id: 'snap_1',
      },
    });

    const withContext = setSelectedContext(base, {
      resultId: 'r1',
      eventId: 'E1',
      frameId: 'v1_10',
    });

    const turned = startFeedbackTurn(withContext, {
      requestId: 'req_1',
      message: 'Make it a blue dress',
    });

    expect(turned.status).toBe('pending');
    expect(turned.pendingRequestId).toBe('req_1');
    expect(turned.messages).toHaveLength(2); // 1 initial assistant + 1 user
    expect(turned.messages[1]).toEqual(
      expect.objectContaining({
        role: 'user',
        text: 'Make it a blue dress',
        context: { resultId: 'r1', eventId: 'E1', frameId: 'v1_10' },
      }),
    );
  });

  test('clarification turn does not replace existing results or advance revision', () => {
    const base = initFeedbackSession(createInitialFeedbackSession(), {
      sessionId: 'fbs_123',
      feedbackRevision: 1,
      state: {
        session_id: 'fbs_123',
        status: 'applied',
        feedback_revision: 1,
        intent: SAMPLE_INTENT,
        results: INITIAL_RESULTS,
        evidence_snapshot_id: 'snap_1',
      },
    });

    const pending = startFeedbackTurn(base, {
      requestId: 'req_clarify',
      message: 'blue or red',
    });

    const clarified = applyFeedbackSuccess(
      pending,
      {
        session_id: 'fbs_123',
        status: 'clarification',
        feedback_revision: 1,
        assistant_message: 'Did you mean dark blue or light blue?',
        intent: SAMPLE_INTENT,
        results: [], // clarification may have empty results list
      },
      { requestId: 'req_clarify' },
    );

    expect(clarified.status).toBe('clarification');
    expect(clarified.feedbackRevision).toBe(1);
    expect(clarified.results).toEqual(INITIAL_RESULTS); // Preserved!
    expect(clarified.messages[clarified.messages.length - 1].text).toBe(
      'Did you mean dark blue or light blue?',
    );
  });

  test('failed turn retains previous results and current intent', () => {
    const base = initFeedbackSession(createInitialFeedbackSession(), {
      sessionId: 'fbs_123',
      feedbackRevision: 1,
      state: {
        session_id: 'fbs_123',
        status: 'applied',
        feedback_revision: 1,
        intent: SAMPLE_INTENT,
        results: INITIAL_RESULTS,
        evidence_snapshot_id: 'snap_1',
      },
    });

    const pending = startFeedbackTurn(base, {
      requestId: 'req_fail',
      message: 'crash server',
    });

    const failed = applyFeedbackFailure(pending, new Error('Network timeout'), {
      requestId: 'req_fail',
    });

    expect(failed.status).toBe('error');
    expect(failed.error).toMatch(/Network timeout/);
    expect(failed.results).toEqual(INITIAL_RESULTS);
    expect(failed.currentIntent).toEqual(SAMPLE_INTENT);
    expect(failed.pendingRequestId).toBeNull();
  });

  test('stale asynchronous response cannot overwrite newer state', () => {
    const base = initFeedbackSession(createInitialFeedbackSession(), {
      sessionId: 'fbs_123',
      feedbackRevision: 2,
      state: {
        session_id: 'fbs_123',
        status: 'applied',
        feedback_revision: 2,
        intent: SAMPLE_INTENT,
        results: INITIAL_RESULTS,
        evidence_snapshot_id: 'snap_1',
      },
    });

    // An older turn 1 response arrives while we are on revision 2
    const ignored = applyFeedbackSuccess(
      base,
      {
        session_id: 'fbs_123',
        status: 'applied',
        feedback_revision: 1,
        intent: SAMPLE_INTENT,
        results: [{ result_id: 'stale_r' }],
      },
      { requestId: 'older_req' },
    );

    expect(ignored.feedbackRevision).toBe(2);
    expect(ignored.results).toEqual(INITIAL_RESULTS);
  });

  test('Undo restores semantic intent, overrides, and previous results with fresh revision', () => {
    const initial = initFeedbackSession(createInitialFeedbackSession(), {
      sessionId: 'fbs_123',
      feedbackRevision: 1,
      state: {
        session_id: 'fbs_123',
        status: 'applied',
        feedback_revision: 1,
        intent: SAMPLE_INTENT,
        results: INITIAL_RESULTS,
        evidence_snapshot_id: 'snap_1',
        can_undo: false,
      },
    });

    const newResults = [
      { result_id: 'r2', video_id: 'v2', frame_idx: 30, timestamp_ms: 3000, score: 0.95 },
    ];

    const turn1 = applyFeedbackSuccess(
      startFeedbackTurn(initial, { requestId: 'req_1', message: 'blue dress' }),
      {
        session_id: 'fbs_123',
        status: 'applied',
        feedback_revision: 2,
        intent: { ...SAMPLE_INTENT, query_text: 'A woman in blue dress' },
        retrieval_overrides: { E1: { dense_text: 'blue dress', bm25_text: 'blue dress' } },
        results: newResults,
        evidence_snapshot_id: 'snap_2',
        can_undo: true,
      },
      { requestId: 'req_1' },
    );

    expect(turn1.feedbackRevision).toBe(2);
    expect(turn1.canUndo).toBe(true);

    // Now Undo
    const undone = applyUndoSuccess(turn1, {
      session_id: 'fbs_123',
      status: 'applied',
      feedback_revision: 3, // Fresh revision after undo
      intent: SAMPLE_INTENT,
      retrieval_overrides: {},
      results: INITIAL_RESULTS,
      evidence_snapshot_id: 'snap_1',
      can_undo: false,
    });

    expect(undone.feedbackRevision).toBe(3);
    expect(undone.currentIntent).toEqual(SAMPLE_INTENT);
    expect(undone.retrievalOverrides).toEqual({});
    expect(undone.results).toEqual(INITIAL_RESULTS);
    expect(undone.canUndo).toBe(false);
  });
});
