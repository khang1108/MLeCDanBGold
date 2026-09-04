/** Resolve a manually entered video moment to canonical frame inspection data. */

import { requestJson } from './client';


const hasNonNegativeInteger = (value) => Number.isSafeInteger(value) && value >= 0;


const validateFrameInspection = (payload) => {
  if (
    !payload
    || typeof payload !== 'object'
    || typeof payload.frame_id !== 'string'
    || !payload.frame_id.trim()
    || typeof payload.video_id !== 'string'
    || !payload.video_id.trim()
    || !hasNonNegativeInteger(payload.requested_timestamp_ms)
    || !hasNonNegativeInteger(payload.frame_idx)
    || !hasNonNegativeInteger(payload.timestamp_ms)
    || !payload.metadata
    || typeof payload.metadata !== 'object'
    || Array.isArray(payload.metadata)
  ) {
    throw new Error('Frame resolver returned an invalid response contract');
  }
  return payload;
};


/** Fetch canonical frame identity and source evidence for a video timestamp. */
export const resolveFrameAtTimestamp = async ({ videoId, timestampMs, signal } = {}) => {
  const normalizedVideoId = String(videoId || '').trim();
  if (!normalizedVideoId) {
    throw new Error('videoId must be a non-blank string');
  }
  if (!hasNonNegativeInteger(timestampMs)) {
    throw new Error('timestampMs must be a non-negative integer');
  }

  const query = new URLSearchParams({
    video_id: normalizedVideoId,
    timestamp_ms: String(timestampMs),
  });
  const payload = await requestJson(`/api/v1/frames/resolve?${query}`, { signal });
  return validateFrameInspection(payload);
};
