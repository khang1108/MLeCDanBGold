import { keyframeUrl } from './keyframes';
import { API_BASE_URL } from './client';

test('builds an absolute keyframe URL and encodes the canonical frame ID', () => {
  expect(keyframeUrl('frame id/1')).toBe(
    `${API_BASE_URL}/api/v1/keyframes/frame%20id%2F1`,
  );
});
