import {
  MOCK_FPS_BY_COLLECTION,
  MOCK_FRAME_FRACTIONS,
  mockFpsForVideo,
  mockSearchFrames,
  mockVideoKeyframes,
} from './mockSearch';

test('returns five media-info videos with multiple canonical frames', () => {
  const payload = mockSearchFrames({ query: 'red boat', topK: 20 });

  expect(payload.results).toHaveLength(15);
  expect(new Set(payload.results.map((frame) => frame.video_id)).size).toBe(5);
  expect(payload.results.filter((frame) => frame.video_id === 'L21_V001')).toHaveLength(3);
  expect(payload.results[0]).toEqual(expect.objectContaining({
    frame_id: 'mock/L21_V001/frame-1',
    video_id: 'L21_V001',
    fps: 30,
    thumbnail_url: expect.stringContaining('i.ytimg.com'),
    mock: true,
  }));
});

test('derives frame indices from timestamps and the configured FPS per video', () => {
  const payload = mockSearchFrames({ query: 'test', topK: 5 });

  payload.results.forEach((frame) => {
    expect(frame.fps).toBe(mockFpsForVideo(frame.video_id));
    expect(frame.frame_idx).toBe(Math.round((frame.timestamp_ms / 1000) * frame.fps));
    expect(frame.timestamp_ms).toBeGreaterThan(0);
    expect(frame.timestamp_ms).toBeLessThanOrEqual(
      Number(frame.video_id === 'L21_V001' ? 1262 : Infinity) * 1000,
    );
  });
  expect(MOCK_FRAME_FRACTIONS).toEqual([0.12, 0.47, 0.81]);
});

test('uses the supplied FPS mapping for every mocked video collection', () => {
  expect(MOCK_FPS_BY_COLLECTION).toEqual({
    L21: 30,
    L22: 30,
    L23: 25,
    L24: 25,
    L25: 25,
  });
});

test('honors a small top-k while retaining one frame per mock video', () => {
  expect(mockSearchFrames({ query: 'test', topK: 2 }).results).toHaveLength(5);
  expect(mockSearchFrames({ query: 'test', topK: 7 }).results).toHaveLength(7);
});

test('provides local neighboring keyframes for a mock result', () => {
  const result = mockSearchFrames({ query: 'kis', topK: 15 }).results[0];
  const keyframes = mockVideoKeyframes(result.frame_id);

  expect(keyframes).toHaveLength(3);
  expect(keyframes).toEqual(expect.arrayContaining([
    expect.objectContaining({ video_id: result.video_id, fps: 30 }),
  ]));
});
