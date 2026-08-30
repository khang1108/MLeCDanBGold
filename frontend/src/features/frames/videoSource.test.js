import { getRaw1FpsFrameId, normalizeSubmissionFps } from './videoSource';

test('normalizes source fps to the nearest BTC submission fps', () => {
  expect(normalizeSubmissionFps(24.98)).toBe(25);
  expect(normalizeSubmissionFps(29.97)).toBe(30);
  expect(normalizeSubmissionFps(27.5)).toBe(25);
});

test('rejects invalid source fps', () => {
  expect(normalizeSubmissionFps(undefined)).toBeNull();
  expect(normalizeSubmissionFps(0)).toBeNull();
  expect(normalizeSubmissionFps('not-a-number')).toBeNull();
});

test('maps a timestamp to the deterministic 1fps frame id', () => {
  expect(getRaw1FpsFrameId('L28_V001', 5_000)).toBe(
    'L28_V001_raw1fps_000000005',
  );
});

test('uses the same keyframe for timestamps within one second', () => {
  expect(getRaw1FpsFrameId('L28_V001', 5_999)).toBe(
    'L28_V001_raw1fps_000000005',
  );
});

test('rejects missing, negative, and non-integer timestamps', () => {
  expect(getRaw1FpsFrameId('', 0)).toBeNull();
  expect(getRaw1FpsFrameId('L28_V001', -1)).toBeNull();
  expect(getRaw1FpsFrameId('L28_V001', 1.5)).toBeNull();
});
