import { searchFrames, searchFramesByImage, searchTrake } from './search';

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

afterEach(() => jest.restoreAllMocks());

test('posts the standalone search request with explicit retrieval sources', async () => {
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
        use_dense: true,
        use_bm25: true,
      }),
    }),
  );
  expect(payload.results[0].frame_id).toBe('f1');
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

test('rounds latency stages to 2 decimal places', async () => {
  const payload = {
    query: 'chef cooks',
    events: ['chef cooks'],
    results: [],
    latency: {
      query_ms: 1.23456,
      retrieval_ms: 2.34567,
      alignment_ms: 3.45678,
      materialization_ms: 4.56789,
      total_ms: 11.6049,
    },
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  const result = await searchFrames({ query: 'chef cooks', topK: 10 });
  expect(result.latency).toEqual({
    query_ms: 1.23,
    retrieval_ms: 2.35,
    alignment_ms: 3.46,
    materialization_ms: 4.57,
    total_ms: 11.6,
  });
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
    expect.stringContaining('/api/v1/trake'),
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        events: ['person enters', 'person leaves'],
        top_k: 20,
        use_dense: true,
        use_bm25: true,
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
    expect.stringContaining('/api/v1/trake'),
    expect.objectContaining({
      body: JSON.stringify({
        events: ['only one'],
        top_k: 20,
        use_dense: true,
        use_bm25: true,
      }),
    }),
  );
});

test('forwards a BM25-only selection to both search contracts', async () => {
  const kisPayload = { events: [], results: [], latency: {
    query_ms: 0,
    retrieval_ms: 0,
    alignment_ms: 0,
    materialization_ms: 0,
    total_ms: 0,
  } };
  const trakePayload = { events: ['event'], paths: [], latency: kisPayload.latency };
  jest.spyOn(global, 'fetch')
    .mockResolvedValueOnce(response(kisPayload))
    .mockResolvedValueOnce(response(trakePayload));

  await searchFrames({
    query: 'boat',
    topK: 10,
    useDense: false,
    useBm25: true,
  });
  await searchTrake({
    events: ['event'],
    topK: 10,
    useDense: false,
    useBm25: true,
  });

  expect(global.fetch.mock.calls[0][1].body).toBe(JSON.stringify({
    query: 'boat',
    top_k: 10,
    use_dense: false,
    use_bm25: true,
  }));
  expect(global.fetch.mock.calls[1][1].body).toBe(JSON.stringify({
    events: ['event'],
    top_k: 10,
    use_dense: false,
    use_bm25: true,
  }));
});

test('rejects disabling both retrieval sources before contacting the backend', async () => {
  const fetchSpy = jest.spyOn(global, 'fetch');

  await expect(searchFrames({
    query: 'boat',
    topK: 10,
    useDense: false,
    useBm25: false,
  })).rejects.toThrow('at least one retrieval source');

  expect(fetchSpy).not.toHaveBeenCalled();
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

test('posts multipart image search request and returns normalized results and latency', async () => {
  const payload = {
    results: [{
      frame_id: 'img-f1',
      video_id: 'L01_V001',
      frame_idx: 10,
      timestamp_ms: 1000,
      score: 0.88,
      frame_ids: ['img-f1'],
      timestamps_ms: [1000],
      metadata: { caption: 'Kitchen with red pot' },
    }],
    latency: {
      query_ms: 10.123,
      retrieval_ms: 20.456,
      alignment_ms: 0,
      materialization_ms: 5.789,
      total_ms: 36.368,
    },
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  const fakeFile = new File(['fake content'], 'test.png', { type: 'image/png' });
  const result = await searchFramesByImage({ imageFile: fakeFile, topK: 15 });

  expect(result.results[0].frame_id).toBe('img-f1');
  expect(result.latency.total_ms).toBe(36.37);
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/search/image'),
    expect.objectContaining({
      method: 'POST',
      body: expect.any(FormData),
    }),
  );
});

test('throws if no imageFile is provided to searchFramesByImage', async () => {
  await expect(searchFramesByImage({ imageFile: null, topK: 20 })).rejects.toThrow(
    'An image file is required for image search',
  );
});

test('rejects a malformed image search response', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(
    response({ results: 'not an array' }),
  );

  const fakeFile = new File(['fake content'], 'test.png', { type: 'image/png' });
  await expect(searchFramesByImage({ imageFile: fakeFile, topK: 20 }))
    .rejects.toThrow('Image search server returned an invalid response contract');
});
