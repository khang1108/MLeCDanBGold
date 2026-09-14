import { normalizeSearchLatency, searchFramesByImage } from './search';

const response = (payload, status = 200, headers = {}) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
  headers: { get: jest.fn((name) => headers[name] ?? null) },
});

afterEach(() => jest.restoreAllMocks());

test('rounds latency stages to 2 decimal places', () => {
  const latency = {
    query_ms: 1.23456,
    retrieval_ms: 2.34567,
    alignment_ms: 3.45678,
    materialization_ms: 4.56789,
    total_ms: 11.6049,
  };
  expect(normalizeSearchLatency(latency)).toEqual({
    query_ms: 1.23,
    retrieval_ms: 2.35,
    alignment_ms: 3.46,
    materialization_ms: 4.57,
    total_ms: 11.6,
  });
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
