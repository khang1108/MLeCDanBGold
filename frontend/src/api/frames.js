import { requestJson, resolveApiUrl } from './client';
import { keyframeUrl } from './keyframes';

/**
 * Fetch the additive detail contract for one canonical frame.
 *
 * Filter requests use this endpoint for every frame in the current page. The
 * caller controls when to fetch it; the endpoint itself remains reusable by
 * the shared frame viewer.
 */
export const getFrameDetail = async ({ frameId, signal } = {}) => {
  if (typeof frameId !== 'string' || !frameId.trim()) {
    throw new Error('Frame detail request requires a canonical frame_id');
  }

  const payload = await requestJson(
    `/api/v1/frames/${encodeURIComponent(frameId)}`,
    { signal },
  );
  if (!payload || payload.frame_id !== frameId) {
    throw new Error('Frame detail response changed canonical frame identity');
  }

  const frameUrl = payload.frame_url
    ? resolveApiUrl(payload.frame_url)
    : keyframeUrl(frameId);
  return {
    ...payload,
    ...(payload.timestamp_ms === undefined && payload.timestamp !== undefined
      ? { timestamp_ms: payload.timestamp }
      : {}),
    frame_url: frameUrl,
  };
};
