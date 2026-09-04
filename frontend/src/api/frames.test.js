import { resolveFrameAtTimestamp } from './frames';


const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});


afterEach(() => jest.restoreAllMocks());


test('resolves a manual video timestamp to canonical inspector metadata', async () => {
  const payload = {
    requested_timestamp_ms: 12_000,
    frame_id: 'L21_V001_00000300',
    video_id: 'L21_V001',
    frame_idx: 300,
    timestamp_ms: 12_040,
    fps: 25,
    metadata: {
      title: 'News clip',
      caption: 'A traffic scene',
      ocr: 'HCM CITY',
      objects: ['traffic', 'car'],
      asr: 'Traffic is heavy today.',
    },
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  await expect(resolveFrameAtTimestamp({
    videoId: 'L21_V001',
    timestampMs: 12_000,
  })).resolves.toEqual(payload);

  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringMatching(/\/api\/v1\/frames\/resolve\?video_id=L21_V001&timestamp_ms=12000$/),
    expect.objectContaining({ method: 'GET' }),
  );
});


test('rejects a resolver response without canonical inspector fields', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({ video_id: 'L21_V001' }));

  await expect(resolveFrameAtTimestamp({
    videoId: 'L21_V001',
    timestampMs: 12_000,
  })).rejects.toThrow('Frame resolver returned an invalid response contract');
});
