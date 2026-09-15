import { connectVbsSession, disconnectVbsSession, getVbsSessionStatus } from './vbs';
import { searchFramesByImage } from './search';
import { searchKis } from './kis';
import { filterFrames } from './filter';

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

afterEach(() => jest.restoreAllMocks());

test('connect sends only the participant ID and accepts a safe session status', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    user_id: 'team-a', connected: true, dres_session: 'private', credential: 'private',
  }));

  await expect(connectVbsSession(' team-a ')).resolves.toEqual({ user_id: 'team-a', connected: true });
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/vbs/session/connect'),
    expect.objectContaining({ method: 'POST', body: JSON.stringify({ user_id: 'team-a' }) }),
  );
});

test('status and disconnect address an encoded participant ID without exposing credentials', async () => {
  jest.spyOn(global, 'fetch')
    .mockResolvedValueOnce(response({ user_id: 'team/a', connected: true }))
    .mockResolvedValueOnce(response({ user_id: 'team/a', connected: false }));

  await expect(getVbsSessionStatus('team/a')).resolves.toEqual({ user_id: 'team/a', connected: true });
  await expect(disconnectVbsSession('team/a')).resolves.toEqual({ user_id: 'team/a', connected: false });
  expect(global.fetch.mock.calls[0][0]).toContain('/api/v1/vbs/session/team%2Fa');
  expect(global.fetch.mock.calls[1][0]).toContain('/api/v1/vbs/session/team%2Fa');
  expect(global.fetch.mock.calls[1][1].method).toBe('DELETE');
  expect(JSON.stringify(global.fetch.mock.calls)).not.toMatch(/credential|password|token|session_id/i);
});

test('text and image search attach only a connected participant ID', async () => {
  const searchResponse = {
    intent: { revision: 1, inputs: ['boat'], events: [{ id: 'E1', text: 'boat' }] },
    operation_summary: { operation_kind: 'initial_resolve', description: 'initial' },
    results: [],
    latency: { total_ms: 1 },
  };
  jest.spyOn(global, 'fetch')
    .mockResolvedValueOnce(response(searchResponse))
    .mockResolvedValueOnce(response(searchResponse))
    .mockResolvedValueOnce(response({ results: [], latency: { total_ms: 1 } }));

  await searchKis({ operation: { kind: 'initial_resolve', text: 'boat' }, topK: 3, userId: 'team-a' });
  await searchKis({ operation: { kind: 'initial_resolve', text: 'boat' }, topK: 3, userId: '' });
  const file = new File(['image'], 'boat.png', { type: 'image/png' });
  await searchFramesByImage({ imageFile: file, topK: 3, userId: 'team-a' });

  expect(global.fetch.mock.calls[0][1].headers['X-VBS-User-ID']).toBe('team-a');
  expect(global.fetch.mock.calls[1][1].headers['X-VBS-User-ID']).toBeUndefined();
  expect(global.fetch.mock.calls[2][1].headers['X-VBS-User-ID']).toBe('team-a');
});

test('Filter attaches the connected participant ID and omits it while disconnected', async () => {
  jest.spyOn(global, 'fetch')
    .mockResolvedValueOnce(response({ frames_per_pages: 20 }))
    .mockResolvedValueOnce(response({ frames_per_pages: 20 }));

  await filterFrames({ filters: {}, userId: 'team-a' });
  await filterFrames({ filters: {}, userId: ' ' });

  expect(global.fetch.mock.calls[0][1].headers['X-VBS-User-ID']).toBe('team-a');
  expect(global.fetch.mock.calls[1][1].headers['X-VBS-User-ID']).toBeUndefined();
});
