import {
  frameAssetUrl,
  searchFrames,
  searchTrake,
} from './search';

const originalMockBackend = process.env.REACT_APP_MOCK_BACKEND;

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

beforeEach(() => {
  delete process.env.REACT_APP_MOCK_BACKEND;
});

afterEach(() => jest.restoreAllMocks());

afterAll(() => {
  if (originalMockBackend === undefined) delete process.env.REACT_APP_MOCK_BACKEND;
  else process.env.REACT_APP_MOCK_BACKEND = originalMockBackend;
});

test('builds canonical frame asset URLs for materialized TRAKE cards', () => {
  expect(frameAssetUrl('folder/frame 1', 'thumbnail')).toBe(
    'http://127.0.0.1:8000/api/v1/frames/folder%2Fframe%201/thumbnail',
  );
  expect(frameAssetUrl('folder/frame 1', 'image')).toBe(
    'http://127.0.0.1:8000/api/v1/frames/folder%2Fframe%201/image',
  );
});

test('posts the canonical standalone search request', async () => {
  const payload = { results: [], latency_ms: { total: 1 } };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  await expect(searchFrames({
    query: ' red boat ',
    topK: 20,
    queryType: 'kis',
  })).resolves.toEqual(payload);

  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/search',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        query: 'red boat',
        top_k: 20,
        query_type: 'kis',
      }),
    }),
  );
});

test('uses the local KIS mock only when the mock env flag is enabled', async () => {
  const previous = process.env.REACT_APP_MOCK_BACKEND;
  process.env.REACT_APP_MOCK_BACKEND = 'true';
  const fetchSpy = jest.spyOn(global, 'fetch');

  try {
    const payload = await searchFrames({ query: 'red boat', topK: 5, queryType: 'kis' });
    expect(payload.results).toHaveLength(5);
    expect(fetchSpy).not.toHaveBeenCalled();
  } finally {
    if (previous === undefined) delete process.env.REACT_APP_MOCK_BACKEND;
    else process.env.REACT_APP_MOCK_BACKEND = previous;
  }
});

test('resolves API-relative frame asset URLs', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    results: [{
      frame_id: 'f1',
      video_id: 'L21_a_b.folder2.L21_V001',
      frame_idx: 125,
      fps: 25,
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
  expect(payload.results[0]).toEqual(expect.objectContaining({
    video_id: 'L21_a_b.folder2.L21_V001',
    frame_idx: 125,
    fps: 25,
  }));
});

test('builds thumbnail identity from frame_id instead of frame_idx or server URL', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    results: [{
      frame_id: 'internal-frame-7',
      video_id: 'L21_V001',
      frame_idx: 900,
      fps: 30,
      timestamp_ms: 30_000,
      thumbnail_url: '/api/v1/frames/900/thumbnail',
      frame_url: '/api/v1/frames/900/image',
    }],
    latency_ms: { total: 1 },
  }));

  const payload = await searchFrames({ query: 'frame identity', topK: 1 });

  expect(payload.results[0].thumbnail_url).toBe(
    'http://127.0.0.1:8000/api/v1/frames/internal-frame-7/thumbnail',
  );
  expect(payload.results[0].frame_url).toBe(
    'http://127.0.0.1:8000/api/v1/frames/internal-frame-7/image',
  );
});

test('rejects a successful frame response without internal frame_id', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    results: [{ video_id: 'L21_V001', frame_idx: 12 }],
    latency_ms: { total: 1 },
  }));

  await expect(searchFrames({ query: 'missing identity', topK: 1 }))
    .rejects.toThrow('missing canonical frame_id');
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

test('sends a progressive search ID when resuming', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    results: [], latency_ms: { total: 1 }, search_id: 'search-1',
  }));
  await searchFrames({
    query: 'H1 H2', topK: 20, queryType: 'kis', searchId: 'search-1',
  });
  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/search',
    expect.objectContaining({
      body: JSON.stringify({
        query: 'H1 H2', top_k: 20, query_type: 'kis', search_id: 'search-1',
      }),
    }),
  );
});

test('posts explicit ordered events to the dedicated TRAKE route', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    events: ['person enters', 'person leaves'],
    submissions: [],
    total_results: 0,
  }));

  await searchTrake({
    events: [' person enters ', ' person leaves '],
    topK: 20,
  });

  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/trake',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        query_type: 'trake',
        query: 'E1: person enters\nE2: person leaves',
        events: ['person enters', 'person leaves'],
        top_k: 20,
      }),
    }),
  );
});

test('rejects fewer than two TRAKE events before contacting the backend', async () => {
  const fetchSpy = jest.spyOn(global, 'fetch');
  await expect(searchTrake({ events: ['only one'], topK: 20 }))
    .rejects.toThrow('at least two');
  expect(fetchSpy).not.toHaveBeenCalled();
});

test('backend network failures stay visible and never produce fake results', async () => {
  jest.spyOn(global, 'fetch').mockRejectedValue(new TypeError('offline'));

  await expect(searchFrames({ query: 'boat', topK: 20 }))
    .rejects.toThrow('Could not reach the backend: offline');
  expect(global.fetch).toHaveBeenCalledTimes(1);
});

test('malformed TRAKE requests are not retried', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(
    response({ detail: [{ msg: 'events must contain at least 2 items' }] }, 422),
  );

  await expect(searchTrake({ events: ['one', 'two'], topK: 20 }))
    .rejects.toThrow('events must contain at least 2 items');
  expect(global.fetch).toHaveBeenCalledTimes(1);
});
