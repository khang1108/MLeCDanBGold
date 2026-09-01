import { keyframeUrl } from './keyframes';

test('builds an absolute keyframe URL and encodes the canonical frame ID', () => {
  expect(keyframeUrl('frame id/1')).toBe(
    'http://127.0.0.1:8000/api/v1/keyframes/frame%20id%2F1',
  );
});
