import { searchKis, uploadKisImage, kisImageAssetUrl } from './kis';

const mockResponse = (payload, status = 200, headers = {}) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
  headers: { get: jest.fn((name) => headers[name] ?? null) },
});

afterEach(() => jest.restoreAllMocks());

test('posts revisioned KIS search request with base_intent and operation to /api/v1/kis/search', async () => {
  const payload = {
    intent: {
      revision: 1,
      query_text: 'chef cooks',
      entities: [],
      events: [{ id: 'E1', text: 'chef cooks' }],
      temporal_edges: [],
    },
    operation_summary: {
      kind: 'initial_resolve',
      affected_event_ids: ['E1'],
    },
    evidence_snapshot_id: 'snap_1',
    results: [{
      result_id: 'r_1',
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

  const operation = {
    kind: 'initial_resolve',
    text: 'chef cooks',
    image_refs: [],
    patches: [],
  };

  const result = await searchKis({
    baseIntent: null,
    expectedRevision: 0,
    operation,
    topK: 20,
    useDense: true,
    useBm25: true,
    userId: 'team-alpha',
  });

  expect(result.intent.query_text).toBe('chef cooks');
  expect(result.evidence_snapshot_id).toBe('snap_1');
  expect(result.operation_summary.kind).toBe('initial_resolve');
  expect(result.results[0].frame_id).toBe('f1');
  expect(result.results[0].result_id).toBe('r_1');
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/kis/search'),
    expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({
        'Content-Type': 'application/json',
        'X-VBS-User-ID': 'team-alpha',
      }),
      body: JSON.stringify({
        base_intent: null,
        expected_revision: 0,
        operation,
        use_dense: true,
        use_bm25: true,
        top_k: 20,
      }),
    }),
  );
});

test('validates operation is required', async () => {
  await expect(searchKis({ operation: null })).rejects.toThrow('Operation is required');
});

test('validates at least one retrieval source', async () => {
  await expect(searchKis({
    operation: { kind: 'search_only' },
    useDense: false,
    useBm25: false,
  })).rejects.toThrow('Enable at least one retrieval source');
});

test('validates response contract', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(mockResponse({
    intent: { revision: 1 },
    // missing operation_summary
    results: [],
    latency: { total_ms: 5 },
  }));

  await expect(searchKis({
    operation: { kind: 'search_only' },
  })).rejects.toThrow('Search server returned an invalid response contract');
});

test('kisImageAssetUrl returns encoded asset URL', () => {
  expect(kisImageAssetUrl('asset/123')).toBe('/api/v1/kis/assets/images/asset%2F123');
});

test('uploadKisImage sends FormData to /api/v1/kis/assets/images', async () => {
  const assetRef = {
    asset_id: 'ast_abc',
    file_name: 'test.jpg',
    content_type: 'image/jpeg',
    byte_size: 1024,
    sha256: 'deadbeef',
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(mockResponse(assetRef, 201));

  const file = new File(['test-bytes'], 'test.jpg', { type: 'image/jpeg' });
  const result = await uploadKisImage({ imageFile: file });

  expect(result.asset_id).toBe('ast_abc');
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/kis/assets/images'),
    expect.objectContaining({
      method: 'POST',
    }),
  );
});
