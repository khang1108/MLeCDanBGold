import { requestJson } from './client';

// One neighbor call with a window wider than any source video returns every
// keyframe of that video, ordered by canonical timestamp.
export const fetchVideoKeyframes = async (frameId, signal) => {
  const payload = await requestJson(
    `/api/v1/frames/${encodeURIComponent(frameId)}/neighbors?window_ms=3600000`,
    { signal },
  );
  if (!Array.isArray(payload)) return [];
  return payload
    .filter((item) => Number.isFinite(item?.timestamp_ms)
      && Number.isInteger(item?.frame_idx))
    .sort((a, b) => a.timestamp_ms - b.timestamp_ms);
};
