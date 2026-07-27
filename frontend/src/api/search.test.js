import { answerFrameQuestion, searchFrames, searchKisc } from './search';

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
  })).resolves.toEqual(payload);

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

test('resolves API-relative frame asset URLs', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue({
    ok: true,
    json: async () => ({
      results: [{
        frame_id: 'f1',
        thumbnail_url: '/api/v1/frames/f1/thumbnail',
        frame_url: '/api/v1/frames/f1/image',
      }],
      latency_ms: { total: 1 },
    }),
  });

  const payload = await searchFrames({
    query: 'red car',
    topK: 1,
    searchMode: 'fast',
  });

  expect(payload.results[0].thumbnail_url).toBe(
    'http://127.0.0.1:8000/api/v1/frames/f1/thumbnail',
  );
  expect(payload.results[0].frame_url).toBe(
    'http://127.0.0.1:8000/api/v1/frames/f1/image',
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

test('posts canonical KISC and VQA requests', async () => {
  const responses = [
    {
      interpreted_state: { standalone_query: 'red car' },
      search: { results: [], latency_ms: { total: 1 } },
    },
    { frame_id: 'f1', answer: 'A red car.' },
  ];
  jest.spyOn(global, 'fetch').mockImplementation(async () => ({
    ok: true,
    json: async () => responses.shift(),
  }));

  await searchKisc({ currentMessage: ' red car ' });
  await answerFrameQuestion({ frameId: 'f1', question: ' What is visible? ' });

  expect(global.fetch.mock.calls[0][0]).toContain('/api/v1/kisc/search');
  expect(JSON.parse(global.fetch.mock.calls[0][1].body).current_message).toBe('red car');
  expect(global.fetch.mock.calls[1][0]).toContain('/api/v1/vqa');
  expect(JSON.parse(global.fetch.mock.calls[1][1].body)).toEqual({
    frame_id: 'f1',
    question: 'What is visible?',
  });
});
