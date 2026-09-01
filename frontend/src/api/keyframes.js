import { API_BASE_URL } from './client';

/** Build the sole public keyframe asset URL from a canonical frame ID. */
export const keyframeUrl = (frameId) => (
  `${API_BASE_URL}/api/v1/keyframes/${encodeURIComponent(frameId)}`
);
