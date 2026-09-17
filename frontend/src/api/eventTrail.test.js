import {
  openEventTrail,
  getEventTrail,
  actOnEventTrail,
  closeEventTrail,
} from './eventTrail';

describe('eventTrail API transport', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  const validState = {
    session_id: 'trail_1',
    result_id: 'r_1',
    video_id: 'V01',
    kis_revision: 3,
    trail_revision: 0,
    status: 'active',
    path: [
      {
        event_id: 'E1',
        frame_id: 'f1',
        frame_idx: 100,
        timestamp_ms: 1000,
        keyframe_path: 'keyframes/f1.jpg',
      },
    ],
    last_valid_path: null,
    approved_event_ids: [],
    rejected_counts: {},
    window: null,
    submission_selection: null,
    transition: null,
  };

  test('openEventTrail serializes payload correctly and validates response', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: { get: () => null },
      json: async () => validState,
    });

    const res = await openEventTrail({
      snapshotId: 'snap_1',
      resultId: 'r_1',
      expectedKisRevision: 3,
      searchSessionId: 'query_1',
    });

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/v1\/event-trail\/open$/),
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          snapshot_id: 'snap_1',
          result_id: 'r_1',
          expected_kis_revision: 3,
          search_session_id: 'query_1',
        }),
      }),
    );
    expect(res).toEqual(validState);
  });

  test('openEventTrail rejects malformed response missing session_id or trail_revision', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: { get: () => null },
      json: async () => ({ session_id: 'trail_1' }), // missing trail_revision and status
    });

    await expect(openEventTrail({
      snapshotId: 'snap_1',
      resultId: 'r_1',
      expectedKisRevision: 3,
    })).rejects.toThrow(/malformed|invalid/i);
  });

  test('getEventTrail queries session status', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: { get: () => null },
      json: async () => validState,
    });

    const res = await getEventTrail('trail_1');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/v1\/event-trail\/trail_1$/),
      expect.objectContaining({ method: 'GET' }),
    );
    expect(res).toEqual(validState);
  });

  test('actOnEventTrail serializes action request body correctly', async () => {
    const nextState = { ...validState, trail_revision: 1 };
    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: { get: () => null },
      json: async () => nextState,
    });

    const res = await actOnEventTrail('trail_1', {
      expectedTrailRevision: 0,
      action: { type: 'approve', event_id: 'E1' },
    });

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/v1\/event-trail\/trail_1\/actions$/),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          expected_trail_revision: 0,
          action: { type: 'approve', event_id: 'E1' },
        }),
      }),
    );
    expect(res.trail_revision).toBe(1);
  });

  test('closeEventTrail calls DELETE with expected_trail_revision query parameter', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 204,
      headers: { get: () => null },
      json: async () => null,
    });

    await closeEventTrail('trail_1', 2);

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/v1\/event-trail\/trail_1\?expected_trail_revision=2$/),
      expect.objectContaining({ method: 'DELETE' }),
    );
  });
});
