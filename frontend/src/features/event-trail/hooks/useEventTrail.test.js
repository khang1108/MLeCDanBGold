import { act, renderHook } from '@testing-library/react';
import {
  openEventTrail,
  getEventTrail,
  actOnEventTrail,
  closeEventTrail,
} from '../../../api/eventTrail';
import { useEventTrail, computeSessionKey } from './useEventTrail';

jest.mock('../../../api/eventTrail', () => ({
  openEventTrail: jest.fn(),
  getEventTrail: jest.fn(),
  actOnEventTrail: jest.fn(),
  closeEventTrail: jest.fn(),
}));

const makeSession = (revision = 0, status = 'active', overrides = {}) => ({
  session_id: 'trail_1',
  result_id: 'r_1',
  video_id: 'V01',
  kis_revision: 2,
  trail_revision: revision,
  status,
  path: [
    { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 1000 },
    { event_id: 'E2', frame_id: 'f2', frame_idx: 20, timestamp_ms: 2000 },
  ],
  last_valid_path: [
    { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 1000 },
    { event_id: 'E2', frame_id: 'f2', frame_idx: 20, timestamp_ms: 2000 },
  ],
  approved_event_ids: [],
  rejected_counts: { E1: 0, E2: 0 },
  window: null,
  submission_selection: null,
  transition: null,
  ...overrides,
});

const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

describe('useEventTrail', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    closeEventTrail.mockResolvedValue();
  });

  test('computeSessionKey produces deterministic string key', () => {
    expect(computeSessionKey({ snapshotId: 's1', resultId: 'r1', kisRevision: 1 })).toBe(
      JSON.stringify(['s1', 'r1', 1])
    );
    expect(computeSessionKey(null)).toBeNull();
  });

  test('open opens a new session and sets state', async () => {
    const mockState = makeSession(0);
    openEventTrail.mockResolvedValueOnce(mockState);

    const { result } = renderHook(() => useEventTrail());

    let res;
    await act(async () => {
      res = await result.current.open({
        snapshotId: 'snap_1',
        resultId: 'r_1',
        kisRevision: 2,
        searchSessionId: 'q_1',
      });
    });

    expect(openEventTrail).toHaveBeenCalledWith(
      {
        snapshotId: 'snap_1',
        resultId: 'r_1',
        expectedKisRevision: 2,
        searchSessionId: 'q_1',
      },
      expect.any(Object)
    );
    expect(res).toEqual(mockState);
    expect(result.current.session).toEqual(mockState);
    expect(result.current.sessionKey).toBe(JSON.stringify(['snap_1', 'r_1', 2]));
    expect(result.current.error).toBeNull();
    expect(result.current.pending).toBe(false);
  });

  test('open reuses active session when called with the same key', async () => {
    const mockState = makeSession(0);
    openEventTrail.mockResolvedValueOnce(mockState);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({
        snapshotId: 'snap_1',
        resultId: 'r_1',
        kisRevision: 2,
        searchSessionId: 'q_1',
      });
    });
    expect(openEventTrail).toHaveBeenCalledTimes(1);

    await act(async () => {
      const reused = await result.current.open({
        snapshotId: 'snap_1',
        resultId: 'r_1',
        kisRevision: 2,
        searchSessionId: 'q_1',
      });
      expect(reused).toEqual(mockState);
    });
    expect(openEventTrail).toHaveBeenCalledTimes(1);
  });

  test('open closes prior session when opening a different key', async () => {
    const sessionA = makeSession(3, 'active', { session_id: 'trail_A' });
    const sessionB = makeSession(0, 'active', { session_id: 'trail_B', result_id: 'r_2' });
    openEventTrail.mockResolvedValueOnce(sessionA).mockResolvedValueOnce(sessionB);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({
        snapshotId: 'snap_1',
        resultId: 'r_1',
        kisRevision: 2,
      });
    });

    await act(async () => {
      await result.current.open({
        snapshotId: 'snap_1',
        resultId: 'r_2',
        kisRevision: 2,
      });
    });

    expect(closeEventTrail).toHaveBeenCalledWith('trail_A', { expectedTrailRevision: 3 });
    expect(result.current.session).toEqual(sessionB);
  });

  test('act applies mutation with expected trail revision', async () => {
    const session0 = makeSession(0);
    const session1 = makeSession(1, 'active', { approved_event_ids: ['E1'] });
    openEventTrail.mockResolvedValueOnce(session0);
    actOnEventTrail.mockResolvedValueOnce(session1);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({ snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 2 });
    });

    let actRes;
    await act(async () => {
      actRes = await result.current.act({ type: 'approve', event_id: 'E1' });
    });

    expect(actOnEventTrail).toHaveBeenCalledWith(
      'trail_1',
      {
        expectedTrailRevision: 0,
        action: { type: 'approve', event_id: 'E1' },
      },
      expect.any(Object)
    );
    expect(actRes).toEqual(session1);
    expect(result.current.session.trail_revision).toBe(1);
  });

  test('act is blocked when another request is pending', async () => {
    const session0 = makeSession(0);
    const d = deferred();
    openEventTrail.mockResolvedValueOnce(session0);
    actOnEventTrail.mockReturnValueOnce(d.promise);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({ snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 2 });
    });

    let firstActPromise;
    act(() => {
      firstActPromise = result.current.act({ type: 'approve', event_id: 'E1' });
    });
    expect(result.current.pending).toBe(true);

    let secondActResult;
    await act(async () => {
      secondActResult = await result.current.act({ type: 'decline', event_id: 'E2' });
    });
    expect(secondActResult).toBeNull();
    expect(actOnEventTrail).toHaveBeenCalledTimes(1);

    await act(async () => {
      d.resolve(makeSession(1));
      await firstActPromise;
    });
    expect(result.current.pending).toBe(false);
  });

  test('TRAIL_REVISION_CONFLICT triggers GET refresh without replay', async () => {
    const session0 = makeSession(0);
    const refreshed = makeSession(2);
    openEventTrail.mockResolvedValueOnce(session0);

    const conflictError = new Error('Revision mismatch');
    conflictError.code = 'TRAIL_REVISION_CONFLICT';
    conflictError.status = 409;
    actOnEventTrail.mockRejectedValueOnce(conflictError);
    getEventTrail.mockResolvedValueOnce(refreshed);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({ snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 2 });
    });

    await act(async () => {
      await result.current.act({ type: 'approve', event_id: 'E1' });
    });

    expect(getEventTrail).toHaveBeenCalledWith('trail_1', expect.any(Object));
    expect(result.current.session).toEqual(refreshed);
    expect(result.current.error).toContain('Conflict detected');
  });

  test('CONSTRAINT_CONFLICT keeps state and displays message without GET refresh', async () => {
    const session0 = makeSession(0);
    openEventTrail.mockResolvedValueOnce(session0);

    const constraintError = new Error('Cannot decline anchored event E1');
    constraintError.code = 'CONSTRAINT_CONFLICT';
    constraintError.status = 409;
    actOnEventTrail.mockRejectedValueOnce(constraintError);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({ snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 2 });
    });

    await act(async () => {
      await result.current.act({ type: 'decline', event_id: 'E1' });
    });

    expect(getEventTrail).not.toHaveBeenCalled();
    expect(result.current.session).toEqual(session0);
    expect(result.current.error).toBe('Cannot decline anchored event E1');
  });

  test('TRAIL_SESSION_EXPIRED clears session and sets prompt', async () => {
    const session0 = makeSession(0);
    openEventTrail.mockResolvedValueOnce(session0);

    const expiredError = new Error('Session expired');
    expiredError.code = 'TRAIL_SESSION_EXPIRED';
    expiredError.status = 410;
    actOnEventTrail.mockRejectedValueOnce(expiredError);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({ snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 2 });
    });

    await act(async () => {
      await result.current.act({ type: 'approve', event_id: 'E1' });
    });

    expect(result.current.session).toBeNull();
    expect(result.current.error).toContain('EventTrail session expired. Reopen it from this result.');
  });

  test('SNAPSHOT_EXPIRED on open sets fresh search prompt', async () => {
    const snapshotExpiredError = new Error('Snapshot expired');
    snapshotExpiredError.code = 'SNAPSHOT_EXPIRED';
    snapshotExpiredError.status = 410;
    openEventTrail.mockRejectedValueOnce(snapshotExpiredError);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({ snapshotId: 'snap_old', resultId: 'r_1', kisRevision: 2 });
    });

    expect(result.current.session).toBeNull();
    expect(result.current.error).toContain(
      'EventTrail snapshot expired. Rerun the current search to create a fresh snapshot.'
    );
  });

  test('late open response from superseded branch is cleaned up and ignored', async () => {
    const slowOpen = deferred();
    const fastOpen = deferred();

    openEventTrail
      .mockReturnValueOnce(slowOpen.promise)
      .mockReturnValueOnce(fastOpen.promise);

    const { result } = renderHook(() => useEventTrail());

    act(() => {
      result.current.open({ snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 2 });
    });

    act(() => {
      result.current.open({ snapshotId: 'snap_1', resultId: 'r_2', kisRevision: 2 });
    });

    const sessionB = makeSession(0, 'active', { session_id: 'trail_B', result_id: 'r_2' });
    await act(async () => {
      fastOpen.resolve(sessionB);
    });
    expect(result.current.session).toEqual(sessionB);

    const sessionA = makeSession(0, 'active', { session_id: 'trail_A', result_id: 'r_1' });
    await act(async () => {
      slowOpen.resolve(sessionA);
    });

    // Session B must NOT be overwritten by late Session A
    expect(result.current.session).toEqual(sessionB);
    expect(closeEventTrail).toHaveBeenCalledWith('trail_A', { expectedTrailRevision: 0 });
  });

  test('undo sends undo action', async () => {
    const session0 = makeSession(1);
    const sessionUndo = makeSession(2);
    openEventTrail.mockResolvedValueOnce(session0);
    actOnEventTrail.mockResolvedValueOnce(sessionUndo);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({ snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 2 });
    });

    await act(async () => {
      await result.current.undo();
    });

    expect(actOnEventTrail).toHaveBeenCalledWith(
      'trail_1',
      {
        expectedTrailRevision: 1,
        action: { type: 'undo' },
      },
      expect.any(Object)
    );
    expect(result.current.session).toEqual(sessionUndo);
  });

  test('close invalidates state and notifies backend', async () => {
    const session = makeSession(3);
    openEventTrail.mockResolvedValueOnce(session);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({ snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 2 });
    });

    await act(async () => {
      await result.current.close();
    });

    expect(closeEventTrail).toHaveBeenCalledWith('trail_1', { expectedTrailRevision: 3 });
    expect(result.current.session).toBeNull();
    expect(result.current.sessionKey).toBeNull();
  });

  test('clearLocal clears state without backend call', async () => {
    const session = makeSession(0);
    openEventTrail.mockResolvedValueOnce(session);

    const { result } = renderHook(() => useEventTrail());

    await act(async () => {
      await result.current.open({ snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 2 });
    });

    act(() => {
      result.current.clearLocal();
    });

    expect(closeEventTrail).not.toHaveBeenCalled();
    expect(result.current.session).toBeNull();
    expect(result.current.sessionKey).toBeNull();
  });
});
