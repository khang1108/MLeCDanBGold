import { buildFilterRequest, filterFrames } from './filter';


const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});


afterEach(() => jest.restoreAllMocks());


test('builds the single-keyword Filter contract without FE normalization', () => {
  expect(buildFilterRequest('  ÁO ĐỎ  ', {
    folderId: 'l21',
    videoId: ' L21_V001 ',
    framesPerPage: 24,
    pageId: 2,
  })).toEqual({
    query: 'ÁO ĐỎ',
    folder_id: 'L21',
    video_id: 'L21_V001',
    frames_per_pages: 24,
    page_id: 2,
  });
});


test('posts the literal request and returns complete result metadata', async () => {
  const payload = {
    page_id: 1,
    frames_per_pages: 12,
    total_pages: 1,
    total_results: 1,
    available_sources: ['caption', 'ocr'],
    results: [{
      frame_id: 'L21_V001_keyframe_000001',
      video_id: 'L21_V001',
      frame_idx: 90,
      timestamp_ms: 3000,
      fps: 30,
      folder_id: 'L21',
      title: null,
      caption: 'A red shirt',
      ocr: 'ÁO ĐỎ',
      objects: { person: 1 },
      asr: null,
      matches: { ocr: 'ÁO ĐỎ' },
    }],
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  const result = await filterFrames({ query: 'ao do' });

  expect(result).toEqual(payload);
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringMatching(/\/api\/v1\/filter$/),
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        query: 'ao do',
        folder_id: null,
        video_id: null,
        frames_per_pages: 12,
        page_id: 1,
      }),
    }),
  );
});
