import { searchFrames } from './search';

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

afterEach(() => jest.restoreAllMocks());

test('posts the canonical standalone search request', async () => {
  const payload = { results: [], latency_ms: { total: 1 } };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  await expect(searchFrames({
    query: ' red boat ',
    topK: 20,
    queryType: 'vkis',
  })).resolves.toEqual(payload);

  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/search',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        query: 'red boat',
        top_k: 20,
        query_type: 'vkis',
      }),
    }),
  );
});

test('resolves API-relative frame asset URLs', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    results: [{
      frame_id: 'f1',
      thumbnail_url: '/api/v1/frames/f1/thumbnail',
      frame_url: '/api/v1/frames/f1/image',
    }],
    latency_ms: { total: 1 },
  }));

  const payload = await searchFrames({
    query: 'red car',
    topK: 1,
  });

  expect(payload.results[0].thumbnail_url).toBe(
    'http://127.0.0.1:8000/api/v1/frames/f1/thumbnail',
  );
  expect(payload.results[0].frame_url).toBe(
    'http://127.0.0.1:8000/api/v1/frames/f1/image',
  );
});

test('surfaces the backend error message', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(
    response({ detail: 'Search engine is not initialized' }, 503),
  );

  await expect(searchFrames({
    query: 'red car',
    topK: 20,
  })).rejects.toThrow('Search engine is not initialized');
});

test('rejects a malformed successful search response', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(
    response({ latency_ms: { total: 1 } }),
  );

  await expect(searchFrames({
    query: 'boat',
    topK: 20,
  })).rejects.toThrow('invalid response contract');
});
