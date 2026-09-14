import { buildFilterRequest, filterFrames, serializeObjectFilters } from './filter';


const response = (payload, status = 200, headers = {}) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
  headers: { get: jest.fn((name) => headers[name] ?? null) },
});


afterEach(() => jest.restoreAllMocks());


test('builds separate evidence predicates with backend-owned folder and video scope', () => {
  expect(buildFilterRequest({
    title: '  Bản tin ',
    asr: 'xin chào',
    caption: '',
    ocr: 'BIỂN BÁO',
    objects: [
      { id: 'first', value: 'Person: 1' },
      { id: 'second', value: 'person: 3' },
      { id: 'invalid', value: 'car: lots' },
    ],
  }, {
    folderId: 'l21',
    videoId: ' L21_V001 ',
    pageId: 2,
  })).toEqual({
    metadata_filters: {
      title: 'Bản tin',
      asr: 'xin chào',
      caption: null,
      ocr: 'BIỂN BÁO',
      objects: { person: 3 },
    },
    folder_id: 'L21',
    video_id: 'L21_V001',
    frames_per_pages: 20,
    page_id: 2,
  });
});


test('keeps only the strictest threshold for repeated object labels', () => {
  expect(serializeObjectFilters([
    { id: 'first', value: 'car: 2' },
    { id: 'second', value: 'CAR: 4' },
  ])).toEqual({ car: 4 });
});


test('posts the fixed-size literal request and returns complete result metadata', async () => {
  const payload = {
    page_id: 1,
    frames_per_pages: 20,
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

  const result = await filterFrames({
    filters: { caption: 'ao do' },
    folderId: 'L21',
    videoId: 'L21_V001',
  });

  expect(result).toEqual(payload);
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringMatching(/\/api\/v1\/filter$/),
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        metadata_filters: {
          title: null,
          asr: null,
          caption: 'ao do',
          ocr: null,
          objects: {},
        },
        folder_id: 'L21',
        video_id: 'L21_V001',
        frames_per_pages: 20,
        page_id: 1,
      }),
    }),
  );
});


test('rejects a backend response that violates the fixed page size', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({ frames_per_pages: 12 }));

  await expect(filterFrames()).rejects.toThrow('page size other than 20');
});

test('publishes the DRES log failure status from Filter without changing its response', async () => {
  const payload = {
    page_id: 1,
    frames_per_page: 20,
    frames_per_pages: 20,
    total_pages: 1,
    total_results: 0,
    results: [],
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload, 200, {
    'X-DRES-Log-Status': 'failed',
  }));
  const listener = jest.fn();
  window.addEventListener('hcmai:dres-log-status', listener);

  await expect(filterFrames({ userId: 'team-a' })).resolves.toEqual(payload);
  expect(listener.mock.calls[0][0].detail).toEqual({ userId: 'team-a', status: 'failed' });
  window.removeEventListener('hcmai:dres-log-status', listener);
});
