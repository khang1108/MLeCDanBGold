import { searchFrames } from './search';

afterEach(() => {
  jest.restoreAllMocks();
});

test('posts the canonical search request and returns the response', async () => {
  const payload = {
    results: [],
    latency_ms: { total: 1 },
  };
  jest.spyOn(global, 'fetch').mockResolvedValue({
    ok: true,
    json: async () => payload,
  });

  await expect(searchFrames({
    query: ' red car ',
    topK: 20,
    searchMode: 'accurate',
  })).resolves.toBe(payload);

  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/search',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        query: 'red car',
        top_k: 20,
        search_mode: 'accurate',
      }),
    }),
  );
});

test('surfaces the backend error message', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue({
    ok: false,
    status: 503,
    json: async () => ({ detail: 'Search engine is not initialized' }),
  });

  await expect(searchFrames({
    query: 'red car',
    topK: 20,
    searchMode: 'fast',
  })).rejects.toThrow('Search engine is not initialized');
});
