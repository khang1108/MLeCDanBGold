import { fetchVideoKeyframes } from './frames';

jest.mock('./client', () => ({ requestJson: jest.fn() }));
jest.mock('./mockSearch', () => ({
  mockBackendEnabled: jest.fn(),
  mockVideoKeyframes: jest.fn(),
}));

const { requestJson } = require('./client');
const { mockBackendEnabled, mockVideoKeyframes } = require('./mockSearch');

beforeEach(() => {
  jest.clearAllMocks();
});

test('uses local keyframes instead of the backend while mock mode is enabled', async () => {
  mockBackendEnabled.mockReturnValue(true);
  mockVideoKeyframes.mockReturnValue([{ frame_id: 'mock/L21_V001/frame-1' }]);

  await expect(fetchVideoKeyframes('mock/L21_V001/frame-1')).resolves.toEqual([
    { frame_id: 'mock/L21_V001/frame-1' },
  ]);
  expect(requestJson).not.toHaveBeenCalled();
});

test('requests canonical neighbors outside mock mode', async () => {
  mockBackendEnabled.mockReturnValue(false);
  requestJson.mockResolvedValue([
    { frame_id: 'later', frame_idx: 200, timestamp_ms: 2_000 },
    { frame_id: 'earlier', frame_idx: 100, timestamp_ms: 1_000 },
  ]);

  await expect(fetchVideoKeyframes('frame/1')).resolves.toEqual([
    { frame_id: 'earlier', frame_idx: 100, timestamp_ms: 1_000 },
    { frame_id: 'later', frame_idx: 200, timestamp_ms: 2_000 },
  ]);
  expect(requestJson).toHaveBeenCalledWith(
    '/api/v1/frames/frame%2F1/neighbors?window_ms=3600000',
    { signal: undefined },
  );
});
