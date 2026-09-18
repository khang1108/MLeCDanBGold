import { searchAvs } from './avs';

const response = (payload) => ({
  ok: true,
  status: 200,
  headers: { get: () => null },
  json: async () => payload,
});

describe('searchAvs', () => {
  afterEach(() => {
    jest.resetAllMocks();
  });

  test('posts AVS text search with optional VBS logging identity', async () => {
    global.fetch = jest.fn().mockResolvedValue(response({
      results: [{
        candidate_id: 'f1', frame_id: 'f1', video_id: 'V1', frame_idx: 1,
        timestamp_ms: 1000, fps: 25, retrieval_rank: 1, retrieval_score: .2,
        metadata: { title: null, caption: null, ocr: null, objects: [], asr: null },
      }],
      latency: { retrieval_ms: 10, coverage_ms: 1, materialization_ms: 2, total_ms: 13 },
      candidate_pool_size: 500,
      deduplicated_candidate_count: 300,
      unique_videos: 1,
      warnings: [],
    }));

    const result = await searchAvs({ query: 'seafood', pageSize: 80, userId: 'team-a' });
    expect(result.results[0].candidate_id).toBe('f1');
    expect(global.fetch.mock.calls[0][1].headers['X-VBS-User-ID']).toBe('team-a');
  });

  test('rejects blank query', async () => {
    await expect(searchAvs({ query: '   ' })).rejects.toThrow('An AVS text query is required');
  });

  test('rejects invalid page size', async () => {
    await expect(searchAvs({ query: 'seafood', pageSize: 0 })).rejects.toThrow('AVS page size must be positive');
  });

  test('rejects malformed response where results is not an array', async () => {
    global.fetch = jest.fn().mockResolvedValue(response({ results: 'not an array' }));
    await expect(searchAvs({ query: 'seafood' })).rejects.toThrow();
  });

  test('rejects candidate where candidate_id does not match frame_id', async () => {
    global.fetch = jest.fn().mockResolvedValue(response({
      results: [{
        candidate_id: 'c1', frame_id: 'f1', video_id: 'V1', frame_idx: 1,
        timestamp_ms: 1000, retrieval_rank: 1,
      }],
      latency: { retrieval_ms: 10, coverage_ms: 1, materialization_ms: 2, total_ms: 13 },
      candidate_pool_size: 1, deduplicated_candidate_count: 1, unique_videos: 1, warnings: [],
    }));
    await expect(searchAvs({ query: 'seafood' })).rejects.toThrow();
  });
});
