import {
  createQueryHistory,
  getQueryHistory,
  markFrameViewed,
} from './history';

jest.mock('./client', () => {
  const actual = jest.requireActual('./client');
  return { ...actual, requestJson: jest.fn() };
});

import { requestJson } from './client';

beforeEach(() => {
  requestJson.mockReset();
});

test('creates the exact minimal history body', async () => {
  requestJson.mockResolvedValueOnce({ ok: true });
  const snapshot = { results: [{ frame_id: 'frame-1', score: 0.9, frame_ids: ['frame-1'] }] };

  await createQueryHistory({
    queryId: 'query-1',
    userId: 'team A',
    queryText: 'a query',
    resultSnapshot: snapshot,
  });

  expect(requestJson).toHaveBeenCalledWith('/api/v1/query-history', expect.objectContaining({
    method: 'POST',
    body: {
      query_id: 'query-1',
      user_id: 'team A',
      query_text: 'a query',
      result_snapshot: snapshot,
    },
  }));
});

test('encodes the user id in the history URL', async () => {
  requestJson.mockResolvedValueOnce({ items: [] });
  await getQueryHistory({ userId: 'team A' });
  expect(requestJson).toHaveBeenCalledWith('/api/v1/query-history?user_id=team%20A', { signal: undefined });
});

test('accepts the KIS history contract with viewed activity only', async () => {
  const item = {
    query_id: 'query-1',
    query_text: 'a boat crosses the scene',
    result_snapshot: { results: [] },
    frame_activity: { viewed_frame_ids: ['frame-1'] },
  };
  requestJson.mockResolvedValueOnce({ items: [item] });

  await expect(getQueryHistory({ userId: 'team-a' })).resolves.toEqual({ items: [item] });
});

test('sends the canonical viewed activity body', async () => {
  requestJson.mockResolvedValueOnce({ ok: true });
  await markFrameViewed({ queryId: 'q/1', frameId: 'frame-1' });

  expect(requestJson).toHaveBeenCalledWith(
    '/api/v1/query-history/q%2F1/viewed-frame',
    { method: 'PATCH', body: { frame_id: 'frame-1' }, signal: undefined },
  );
});

test('rejects invalid values before making a request', async () => {
  await expect(getQueryHistory({ userId: ' ' })).rejects.toThrow('userId');
  await expect(markFrameViewed({ queryId: 'q', frameId: ' ' })).rejects.toThrow('frameId');
  expect(requestJson).not.toHaveBeenCalled();
});
