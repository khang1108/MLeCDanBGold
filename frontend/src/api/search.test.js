import { searchFrames, searchTrake } from './search';

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

afterEach(() => jest.restoreAllMocks());

test('posts the standalone search request with only query and top_k', async () => {
  const payload = {
    query: 'chef cooks',
    events: ['chef cooks'],
    results: [{
      frame_id: 'f1',
      video_id: 'L21_a_b.folder2.L21_V001',
      frame_idx: 125,
      timestamp_ms: 5000,
      score: 0.91,
      frame_ids: ['f0', 'f1'],
      timestamps_ms: [4000, 5000],
      thumbnail_urls: ['/thumbs/f0.jpg', 'https://cdn.example/f1.jpg'],
      frame_url: '/api/v1/frames/f1/image',
      thumbnail_url: 'https://cdn.example/f1-thumb.jpg',
      metadata: { title: 'Kitchen scene' },
    }],
    latency: {
      query_ms: 1,
      retrieval_ms: 2,
      alignment_ms: 3,
      materialization_ms: 4,
      total_ms: 10,
    },
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  await expect(searchFrames({
    query: ' chef cooks ',
    topK: 20,
  })).resolves.toEqual(payload);

  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/search'),
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        query: 'chef cooks',
        top_k: 20,
      }),
    }),
  );
  expect(payload.results[0].frame_url).toBe('/api/v1/frames/f1/image');
  expect(payload.results[0].thumbnail_url).toBe('https://cdn.example/f1-thumb.jpg');
  expect(payload.results[0].thumbnail_urls).toEqual([
    '/thumbs/f0.jpg',
    'https://cdn.example/f1.jpg',
  ]);
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
    response({ events: ['boat'], results: [] }),
  );

  await expect(searchFrames({
    query: 'boat',
    topK: 20,
  })).rejects.toThrow('invalid response contract');
});

test('posts explicit ordered events to the dedicated TRAKE route', async () => {
  const payload = {
    events: ['person enters', 'person leaves'],
    paths: [{
      video_id: 'L21_V001',
      score: 2.3,
      frame_ids: ['f0', 'f1'],
      frame_idxs: [100, 140],
      timestamps_ms: [4000, 5600],
      thumbnail_urls: ['/thumbs/f0.jpg', '/thumbs/f1.jpg'],
    }],
    latency: {
      query_ms: 1,
      retrieval_ms: 2,
      alignment_ms: 3,
      materialization_ms: 4,
      total_ms: 10,
    },
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    ...payload,
  }));

  await expect(searchTrake({
    events: [' person enters ', ' person leaves '],
    topK: 20,
  })).resolves.toEqual(payload);

  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/trake',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        events: ['person enters', 'person leaves'],
        top_k: 20,
      }),
    }),
  );
  expect(payload.paths).toHaveLength(1);
  expect(payload.events).toEqual(['person enters', 'person leaves']);
  expect(payload.latency.total_ms).toBe(10);
});

test('posts a single TRAKE event accepted by the backend contract', async () => {
  const payload = {
    events: ['only one'],
    paths: [],
    latency: {
      query_ms: 1,
      retrieval_ms: 2,
      alignment_ms: 3,
      materialization_ms: 4,
      total_ms: 10,
    },
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  await expect(searchTrake({ events: [' only one '], topK: 20 }))
    .resolves.toEqual(payload);
  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/trake',
    expect.objectContaining({
      body: JSON.stringify({ events: ['only one'], top_k: 20 }),
    }),
  );
});

test('rejects TRAKE input with no non-empty events before contacting the backend', async () => {
  const fetchSpy = jest.spyOn(global, 'fetch');

  await expect(searchTrake({ events: ['  '], topK: 20 }))
    .rejects.toThrow('at least one');
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

test('rejects a malformed successful TRAKE response', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(
    response({ events: ['e1', 'e2'], submissions: [] }),
  );

  await expect(searchTrake({ events: ['e1', 'e2'], topK: 20 }))
    .rejects.toThrow('invalid response contract');
});
