import { searchFrames } from './search';

const response = (payload, status = 200) => ({ ok: status >= 200 && status < 300, status, json: jest.fn().mockResolvedValue(payload) });
afterEach(() => jest.restoreAllMocks());

test('posts the canonical stateful search request', async () => {
  const payload = { results: [], latency_ms: { total: 1 } };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));
  await expect(searchFrames({ query: ' red boat ', topK: 20, searchMode: 'accurate', sessionId: 'kisc_sess_1234', feedback: { accepted_frame_ids: ['frame_A'], rejected_frame_ids: [] } })).resolves.toBe(payload);
  expect(global.fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/api/v1/search', expect.objectContaining({ method: 'POST', body: JSON.stringify({ query: 'red boat', top_k: 20, search_mode: 'accurate', session_id: 'kisc_sess_1234', feedback: { accepted_frame_ids: ['frame_A'], rejected_frame_ids: [] } }) }));
});

test('rejects a malformed successful search response', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({ latency_ms: { total: 1 } }));
  await expect(searchFrames({ query: 'boat', topK: 20, searchMode: 'fast' })).rejects.toThrow('invalid response contract');
});