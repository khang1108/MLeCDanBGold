const STREAM_API_BASE_URL = 'https://stream.iamphuckhang.dev/api/v1';

export const displayVideoId = (videoId) => {
  const parts = String(videoId || '').split('.').filter(Boolean);
  return parts[parts.length - 1] || 'Unknown video';
};

export const normalizeSubmissionFps = (fps) => {
  const numericFps = Number(fps);
  if (!Number.isFinite(numericFps) || numericFps <= 0) {
    return null;
  }

  return Math.abs(numericFps - 25) <= Math.abs(numericFps - 30) ? 25 : 30;
};

export const getRaw1FpsFrameId = (videoId, timestampMs) => {
  const canonicalVideoId = String(videoId || '').trim();
  const timestamp = Number(timestampMs);
  if (!canonicalVideoId || !Number.isInteger(timestamp) || timestamp < 0) {
    return null;
  }

  const second = Math.floor(timestamp / 1000);
  return `${canonicalVideoId}_raw1fps_${String(second).padStart(9, '0')}`;
};

export const getStreamVideoUrl = (videoId, timestampMs) => {
  const canonicalVideoId = String(videoId || '').trim();
  const timestamp = Number(timestampMs);
  if (!canonicalVideoId || !Number.isInteger(timestamp) || timestamp < 0) {
    return null;
  }

  // The stream service stores videos by organizer leaf ID. Keep the full
  // canonical ID everywhere else and translate only at this external boundary.
  const streamVideoId = displayVideoId(canonicalVideoId);

  // `/play` is an HTML player page. Native <video> needs the raw MP4 stream;
  // the inspector applies timestampMs after that stream reports metadata.
  return `${STREAM_API_BASE_URL}/videos/${encodeURIComponent(streamVideoId)}/stream`;
};
