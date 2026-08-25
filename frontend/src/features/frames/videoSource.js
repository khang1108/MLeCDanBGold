import mediaInfo from './mediaInfo.generated';

export const displayVideoId = (videoId) => {
  const parts = String(videoId || '').split('.').filter(Boolean);
  return parts[parts.length - 1] || 'Unknown video';
};

const youtubeVideoId = (url) => {
  if (!url) return null;

  try {
    const parsed = new URL(url);
    if (parsed.hostname === 'youtu.be') return parsed.pathname.slice(1) || null;
    if (parsed.hostname.endsWith('youtube.com')) {
      if (parsed.pathname === '/watch') return parsed.searchParams.get('v');
      const embedMatch = parsed.pathname.match(/^\/(?:embed|shorts|live)\/([^/?]+)/);
      return embedMatch?.[1] || null;
    }
  } catch {
    return null;
  }

  return null;
};

export const getYouTubeWatchUrl = (videoId) => {
  const canonicalId = String(videoId || '').trim();
  const leafId = displayVideoId(canonicalId);
  return mediaInfo[canonicalId] || mediaInfo[leafId] || null;
};

export const getYouTubeVideoId = (videoIdOrUrl) => youtubeVideoId(
  String(videoIdOrUrl || '').includes('://')
    ? videoIdOrUrl
    : getYouTubeWatchUrl(videoIdOrUrl),
);

export const getYouTubeEmbedUrl = (videoIdOrUrl) => {
  const id = getYouTubeVideoId(videoIdOrUrl);
  if (!id) return null;

  const params = new URLSearchParams({
    autoplay: '0',
    enablejsapi: '1',
    playsinline: '1',
    rel: '0',
    origin: window.location.origin,
  });
  return `https://www.youtube.com/embed/${encodeURIComponent(id)}?${params.toString()}`;
};

export const timestampSeconds = (timestampMs) => {
  const timestamp = Number(timestampMs);
  return Number.isFinite(timestamp) && timestamp >= 0
    ? timestamp / 1000
    : null;
};
