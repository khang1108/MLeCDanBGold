import { buildFilterRequest, filterFrames, normalizeFilterText } from './filter';

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

afterEach(() => jest.restoreAllMocks());

const completeFrame = (overrides = {}) => ({
  frame_id: 'frame-1',
  video_id: 'L21_V001',
  folder_id: 'L21',
  frame_idx: 12,
  timestamp_ms: 480,
  title: null,
  caption: null,
  ocr: null,
  objects: {},
  asr: null,
  ...overrides,
});

test('normalizes Vietnamese text without changing the input object', () => {
  const filters = {
    title: '  CẢNH   có ÁO ĐỎ  ',
    asr: '',
    caption: '',
    ocr: '',
    objects: [
      { name: 'Ghế', count: '2' },
      { name: ' ghe ', count: '2' },
    ],
  };

  expect(normalizeFilterText(filters.title)).toBe('canh co ao do');
  expect(buildFilterRequest(filters)).toEqual({
    metadata_filters: {
      title: 'canh co ao do',
      asr: null,
      caption: null,
      ocr: null,
      objects: { ghe: 4 },
    },
    folder_id: null,
    video_id: null,
    frames_per_pages: 12,
    page_id: 1,
  });
  expect(filters.objects[0].name).toBe('Ghế');
});

test('parses compact object inputs in the name colon count format', () => {
  expect(buildFilterRequest({
    objects: [{ value: ' chair : 4 ' }, { value: 'person: 2' }],
  })).toEqual({
    metadata_filters: {
      title: null,
      asr: null,
      caption: null,
      ocr: null,
      objects: { chair: 4, person: 2 },
    },
    folder_id: null,
    video_id: null,
    frames_per_pages: 12,
    page_id: 1,
  });
});

test('posts one metadata filter request with empty scope values', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    page_id: 1,
    frames_per_pages: 12,
    total_pages: 3,
    total_results: 1,
    results: [completeFrame({
      objects: { chair: 4 },
    })],
  }));

  await filterFrames({
    filters: {
      title: ' Video ',
      asr: 'hello',
      caption: '',
      ocr: 'text',
      objects: [{ name: 'chair', count: '4' }],
    },
  });

  expect(global.fetch).toHaveBeenCalledTimes(1);
  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/filter',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        metadata_filters: {
          title: 'video',
          asr: 'hello',
          caption: null,
          ocr: 'text',
          objects: { chair: 4 },
        },
        folder_id: null,
        video_id: null,
        frames_per_pages: 12,
        page_id: 1,
      }),
    }),
  );
});

test('sends the requested page and accepts a global total_results count', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    page_id: 4,
    frames_per_pages: 24,
    total_pages: 9,
    total_results: 201,
    results: [completeFrame({
      frame_id: 'frame-25',
      video_id: 'L24_V001',
      folder_id: 'L24',
    })],
  }));

  const payload = await filterFrames({
    filters: {},
    framesPerPage: 24,
    pageId: 4,
  });

  expect(payload.total_pages).toBe(9);
  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/filter',
    expect.objectContaining({
      body: JSON.stringify({
        metadata_filters: {
          title: null,
          asr: null,
          caption: null,
          ocr: null,
          objects: {},
        },
        folder_id: null,
        video_id: null,
        frames_per_pages: 24,
        page_id: 4,
      }),
    }),
  );
});

test('sends folder and video scope at the top level', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    page_id: 1,
    frames_per_pages: 12,
    total_pages: 1,
    total_results: 1,
    results: [completeFrame({
      frame_id: 'scoped-frame',
      video_id: 'L26_V001',
      folder_id: 'L26',
      frame_idx: 1,
      timestamp_ms: 40,
    })],
  }));

  await filterFrames({
    filters: {},
    folderId: 'l26',
    videoId: 'L26_V001',
    framesPerPage: 12,
    pageId: 1,
  });

  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/filter',
    expect.objectContaining({
      body: expect.stringContaining('"folder_id":"L26"'),
    }),
  );
  expect(global.fetch.mock.calls[0][1].body).toContain('"video_id":"L26_V001"');
});

test('rejects filter results that do not preserve canonical identity', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    page_id: 1,
    frames_per_pages: 12,
    total_pages: 1,
    total_results: 1,
    results: [{ video_id: 'L21_V001', frame_idx: 12, timestamp_ms: 480 }],
  }));

  await expect(filterFrames({ filters: {} }))
    .rejects.toThrow('canonical frame_id');
});

test('rejects a filter response for a different page than requested', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    page_id: 3,
    frames_per_pages: 12,
    total_pages: 9,
    total_results: 201,
    results: [],
  }));

  await expect(filterFrames({ filters: {}, pageId: 4 }))
    .rejects.toThrow('page_id 3 for requested page 4');
});

test('requires echoed page size, totals, and complete display metadata', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    page_id: 1,
    frames_per_pages: 12,
    total_pages: 1,
    total_results: 1,
    results: [completeFrame({
      title: 'Episode',
      caption: 'A red shirt',
      ocr: 'STOP',
      objects: { person: 3 },
      asr: 'Xin chào',
    })],
  }));

  const payload = await filterFrames({ filters: {} });

  expect(payload.results[0]).toEqual(expect.objectContaining({
    title: 'Episode',
    caption: 'A red shirt',
    ocr: 'STOP',
    objects: { person: 3 },
    asr: 'Xin chào',
  }));
});

test.each([
  { page_id: 1, total_pages: 0, total_results: 0, results: [] },
  {
    page_id: 1,
    frames_per_pages: 12,
    total_pages: 1,
    results: [completeFrame()],
  },
  {
    page_id: 1,
    frames_per_pages: 24,
    total_pages: 1,
    total_results: 1,
    results: [completeFrame()],
  },
  {
    page_id: 1,
    frames_per_pages: 12,
    total_pages: 1,
    total_results: 1,
    results: [completeFrame({ caption: undefined })],
  },
])('rejects incomplete or mismatched complete-page responses', async (payload) => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  await expect(filterFrames({ filters: {} })).rejects.toThrow();
});

test('rejects filesystem paths in Filter result rows', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    page_id: 1,
    frames_per_pages: 12,
    total_pages: 1,
    total_results: 1,
    results: [completeFrame({ image_path: '/private/keyframes/frame.jpg' })],
  }));

  await expect(filterFrames({ filters: {} })).rejects.toThrow('filesystem path');
});
