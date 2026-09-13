import { act, renderHook } from '@testing-library/react';
import {
  actOnExploration,
  getExploration,
  openExploration,
} from '../../../api/exploration';
import {
  buildExplorationOpenBody,
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

beforeEach(() => jest.clearAllMocks());

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
