import { frameAssetUrl, searchFrames, searchTrake, searchVqa } from './search';

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

afterEach(() => jest.restoreAllMocks());

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

test('posts the dedicated competition VQA request', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    submissions: [],
    latency_ms: 0,
  }));

  await searchVqa({
    eventDescription: ' a person reads a sign ',
    question: ' what does it say? ',
    topK: 100,
  });

  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/vqa',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        query_type: 'vqa',
        event_description: 'a person reads a sign',
        question: 'what does it say?',
        top_k: 100,
      }),
    }),
  );
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

test('adds canonical frame asset URLs to VQA submissions', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    submissions: [{
      frame_id: 'frame/1',
      video_id: 'L21_a_b.folder2.L21_V001',
      frame_idx: 125,
      fps: 25,
      caption: 'A person reads a city sign.',
    }],
    latency_ms: 4,
  }));

  const payload = await searchVqa({
    eventDescription: 'a person reads a sign',
    question: 'what does it say?',
    topK: 1,
  });

  expect(payload.submissions[0]).toEqual(expect.objectContaining({
    caption: 'A person reads a city sign.',
    video_id: 'L21_a_b.folder2.L21_V001',
    frame_idx: 125,
    fps: 25,
    thumbnail_url: 'http://127.0.0.1:8000/api/v1/frames/frame%2F1/thumbnail',
    frame_url: 'http://127.0.0.1:8000/api/v1/frames/frame%2F1/image',
  }));
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
        query: 'person enters | person leaves',
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
