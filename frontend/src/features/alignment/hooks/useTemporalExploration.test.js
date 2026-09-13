import { act, renderHook } from '@testing-library/react';
import {
  actOnExploration,
  closeExploration,
  getExploration,
  openExploration,
} from '../../../api/exploration';
import {
  buildExplorationOpenBody,
  EXPLORATION_REQUEST_TIMEOUT_MS,
  useTemporalExploration,
} from './useTemporalExploration';

jest.mock('../../../api/exploration', () => ({
  openExploration: jest.fn(),
  getExploration: jest.fn(),
  actOnExploration: jest.fn(),
  closeExploration: jest.fn(),
}));

const snapshot = {
  query: 'red boat',
  events: ['red boat'],
  dense_events: ['dense red boat'],
  bm25_caption_events: ['caption red boat'],
  use_dense: true,
  use_bm25: true,
};

const envelope = (revision) => ({
  handle: 'branch-1',
  scoring_revision: 'scoring-1',
  view: { revision, event_version: 'event-1', conditions: { window: [0, 30_000] } },
});

const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
};

beforeEach(() => {
  jest.clearAllMocks();
  closeExploration.mockResolvedValue(null);
});

test('maps only the live snapshot and finite whole-video window for open', () => {
  expect(buildExplorationOpenBody({
    snapshot,
    videoId: 'V01',
    durationSeconds: 30.999,
  })).toEqual({
    query: 'red boat',
    events: ['red boat'],
    retrieval_events: ['dense red boat'],
    caption_events: ['caption red boat'],
    use_dense: true,
    use_bm25: true,
    video_id: 'V01',
    window: [0, 30_999],
  });
  expect(buildExplorationOpenBody({ snapshot, videoId: 'V01', durationSeconds: 0 })).toBeNull();
});

test('reconciles a stale mutation with GET instead of replaying the mutation', async () => {
  openExploration.mockResolvedValueOnce(envelope(1));
  const stale = new Error('stale revision');
  stale.status = 409;
  actOnExploration.mockRejectedValueOnce(stale);
  getExploration.mockResolvedValueOnce(envelope(2));
  const { result } = renderHook(() => useTemporalExploration());

  await act(async () => {
    await result.current.open({ snapshot, videoId: 'V01', durationSeconds: 30 });
  });
  await act(async () => {
    await result.current.undo();
  });

  expect(actOnExploration).toHaveBeenCalledWith('branch-1', {
    expected_revision: 1,
    event_version: 'event-1',
    scoring_revision: 'scoring-1',
    action: 'undo',
  }, expect.objectContaining({ signal: expect.any(Object) }));
  expect(getExploration).toHaveBeenCalledWith('branch-1', expect.objectContaining({ signal: expect.any(Object) }));
  expect(result.current.session.view.revision).toBe(2);
});

test('serializes duplicate mutations and ignores an action response after close', async () => {
  openExploration.mockResolvedValueOnce(envelope(1));
  const action = deferred();
  actOnExploration.mockReturnValueOnce(action.promise);
  const { result } = renderHook(() => useTemporalExploration());

  await act(async () => {
    await result.current.open({ snapshot, videoId: 'V01', durationSeconds: 30 });
  });
  act(() => {
    result.current.undo();
    result.current.undo();
  });
  expect(actOnExploration).toHaveBeenCalledTimes(1);

  await act(async () => {
    await result.current.close();
    action.resolve(envelope(2));
    await action.promise;
  });
  expect(result.current.session).toBeNull();
});

test('reconnects the same query and video but closes a prior video branch', async () => {
  openExploration.mockResolvedValueOnce(envelope(1)).mockResolvedValueOnce({
    ...envelope(1), handle: 'branch-2', view: { ...envelope(1).view, video_id: 'V02' },
  });
  const { result } = renderHook(() => useTemporalExploration());

  await act(async () => {
    await result.current.open({ snapshot, videoId: 'V01', durationSeconds: 30 });
    await result.current.open({ snapshot, videoId: 'V01', durationSeconds: 30 });
  });
  expect(openExploration).toHaveBeenCalledTimes(1);

  await act(async () => {
    await result.current.open({ snapshot, videoId: 'V02', durationSeconds: 30 });
  });
  expect(closeExploration).toHaveBeenCalledWith('branch-1', 1);
  expect(result.current.session.handle).toBe('branch-2');
});

test('clears a missing branch returned by reconciliation without leaving a synchronization error', async () => {
  openExploration.mockResolvedValueOnce(envelope(1));
  const stale = new Error('stale');
  stale.status = 409;
  const missing = new Error('gone');
  missing.status = 404;
  actOnExploration.mockRejectedValueOnce(stale);
  getExploration.mockRejectedValueOnce(missing);
  const { result } = renderHook(() => useTemporalExploration());

  await act(async () => {
    await result.current.open({ snapshot, videoId: 'V01', durationSeconds: 30 });
    await result.current.undo();
  });
  expect(result.current.session).toBeNull();
  expect(result.current.error).toBeNull();
});

test('reconciles a timed-out mutation through GET without replaying it', async () => {
  jest.useFakeTimers();
  openExploration.mockResolvedValueOnce(envelope(1));
  actOnExploration.mockReturnValueOnce(new Promise(() => {}));
  getExploration.mockResolvedValueOnce(envelope(2));
  const { result } = renderHook(() => useTemporalExploration());

  await act(async () => {
    await result.current.open({ snapshot, videoId: 'V01', durationSeconds: 30 });
  });
  act(() => { result.current.undo(); });
  await act(async () => { jest.advanceTimersByTime(EXPLORATION_REQUEST_TIMEOUT_MS); });

  expect(actOnExploration).toHaveBeenCalledTimes(1);
  expect(getExploration).toHaveBeenCalledWith('branch-1', expect.any(Object));
  expect(result.current.session.view.revision).toBe(2);
  jest.useRealTimers();
});

test('blocks mutations after failed reconciliation until refresh succeeds', async () => {
  openExploration.mockResolvedValueOnce(envelope(1));
  const stale = new Error('stale');
  stale.status = 409;
  actOnExploration.mockRejectedValueOnce(stale).mockResolvedValueOnce(envelope(2));
  getExploration.mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(envelope(1));
  const { result } = renderHook(() => useTemporalExploration());

  await act(async () => {
    await result.current.open({ snapshot, videoId: 'V01', durationSeconds: 30 });
    await result.current.undo();
  });
  await act(async () => { await result.current.undo(); });
  expect(actOnExploration).toHaveBeenCalledTimes(1);

  await act(async () => { await result.current.refresh(); });
  await act(async () => { await result.current.undo(); });
  expect(actOnExploration).toHaveBeenCalledTimes(2);
});

test('closes a late successful open after its generation has been invalidated', async () => {
  const opening = deferred();
  openExploration.mockReturnValueOnce(opening.promise);
  const { result } = renderHook(() => useTemporalExploration());

  act(() => { result.current.open({ snapshot, videoId: 'V01', durationSeconds: 30 }); });
  await act(async () => { await result.current.close(); });
  await act(async () => {
    opening.resolve(envelope(1));
    await opening.promise;
  });
  expect(closeExploration).toHaveBeenCalledWith('branch-1', 1);
  expect(result.current.session).toBeNull();
});
