import { searchKis } from './kis';

const mockResponse = (payload, status = 200, headers = {}) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
  headers: { get: jest.fn((name) => headers[name] ?? null) },
});

afterEach(() => jest.restoreAllMocks());

test('posts revisioned KIS search request to /api/v1/kis/search', async () => {
  const payload = {
    intent: {
      revision: 1,
      inputs: ['chef cooks'],
      language: 'en',
      query_text: 'chef cooks',
      entities: [],
      events: [{ id: 'E1', text: 'chef cooks' }],
      temporal_edges: [],
    },
    results: [{
      frame_id: 'f1',
      video_id: 'L21_V001',
      frame_idx: 125,
      timestamp_ms: 5000,
      score: 0.91,
      frame_ids: ['f1'],
      timestamps_ms: [5000],
      metadata: {},
    }],
    latency: {
      query_ms: 1,
      retrieval_ms: 2,
      alignment_ms: 3,
      materialization_ms: 4,
      total_ms: 10,
    },
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(mockResponse(payload));

  const result = await searchKis({
    inputs: [{ text: '  chef cooks  ' }],
    expectedRevision: 0,
    topK: 20,
    useDense: true,
    useBm25: true,
    userId: 'team-alpha',
  });

  expect(result.intent.query_text).toBe('chef cooks');
  expect(result.results[0].frame_id).toBe('f1');
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/kis/search'),
    expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({
        'Content-Type': 'application/json',
        'X-VBS-User-ID': 'team-alpha',
      }),
      body: JSON.stringify({
        inputs: [{ text: 'chef cooks' }],
        expected_revision: 0,
        use_dense: true,
        use_bm25: true,
        top_k: 20,
      }),
    }),
  );
});

test('validates inputs non-empty', async () => {
  await expect(searchKis({ inputs: [] })).rejects.toThrow('Inputs must be a non-empty array');
  await expect(searchKis({ inputs: ['   '] })).rejects.toThrow('Input text must not be blank');
});

test('validates at least one retrieval source', async () => {
  await expect(searchKis({
    inputs: ['query'],
    useDense: false,
    useBm25: false,
  })).rejects.toThrow('Enable at least one retrieval source');
});
