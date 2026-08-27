import mockMediaInfo from '../features/frames/mockMediaInfo.generated';

// The local media-info files do not contain BTC mapping FPS. This explicit
// mapping mirrors the supplied collection-level frame rates for UI testing.
export const MOCK_FPS_BY_COLLECTION = Object.freeze({
  L21: 30,
  L22: 30,
  L23: 25,
  L24: 25,
  L25: 25,
});
export const MOCK_FRAME_FRACTIONS = [0.12, 0.47, 0.81];

const mockVideos = Object.values(mockMediaInfo);

export const mockBackendEnabled = () => /^(1|true|yes)$/i.test(
  process.env.REACT_APP_MOCK_BACKEND || '',
);

const durationSeconds = (video) => (
  Number.isFinite(video.length) && video.length > 0 ? video.length : 60
);

export const mockFpsForVideo = (videoId) => (
  MOCK_FPS_BY_COLLECTION[String(videoId || '').split('_')[0]] || 25
);

const buildMockFrame = (video, videoIndex, frameIndex, query) => {
  const timestampMs = Math.round(
    durationSeconds(video) * MOCK_FRAME_FRACTIONS[frameIndex] * 1000,
  );
  const fps = mockFpsForVideo(video.video_id);
  const frameId = `mock/${video.video_id}/frame-${frameIndex + 1}`;
  const finalScore = Number((0.96 - (videoIndex * 0.05) - (frameIndex * 0.025)).toFixed(3));
  return {
    rank: videoIndex * MOCK_FRAME_FRACTIONS.length + frameIndex + 1,
    frame_id: frameId,
    video_id: video.video_id,
    frame_idx: Math.round((timestampMs / 1000) * fps),
    timestamp_ms: timestampMs,
    fps,
    caption: `${video.title} · mock result for “${query}” · ${Math.round(timestampMs / 1000)}s`,
    thumbnail_url: video.thumbnail_url,
    frame_url: video.thumbnail_url,
    scores: { final: finalScore },
    mock: true,
  };
};

// The modal asks for a video's neighboring keyframes. In mock mode those
// neighbors must be local too; otherwise opening a result would call port 8000.
export const mockVideoKeyframes = (frameId) => {
  const match = String(frameId || '').match(/^mock\/(.+)\/frame-\d+$/);
  const videoId = match?.[1];
  const videoIndex = mockVideos.findIndex((video) => video.video_id === videoId);
  if (videoIndex < 0) return [];

  const video = mockVideos[videoIndex];
  return MOCK_FRAME_FRACTIONS.map((_, frameIndex) => (
    buildMockFrame(video, videoIndex, frameIndex, 'KIS preview')
  ));
};

export const mockSearchFrames = ({ query, topK = 20 } = {}) => {
  const allFrames = mockVideos.flatMap((video, videoIndex) => (
    MOCK_FRAME_FRACTIONS.map((_, frameIndex) => (
      buildMockFrame(video, videoIndex, frameIndex, query || 'KIS preview')
    ))
  ));
  const requestedTopK = Number(topK);
  const resultCount = Number.isFinite(requestedTopK) && requestedTopK > 0
    ? Math.max(mockVideos.length, Math.min(allFrames.length, requestedTopK))
    : allFrames.length;

  return {
    search_id: 'mock-kis-v1',
    results: allFrames.slice(0, resultCount),
    warnings: [
      'MOCK BACKEND enabled: results are generated from local media-info for UI testing.',
      'Mock frame mapping uses the configured collection FPS because media-info does not include BTC FPS mappings.',
    ],
    latency_ms: {
      total: 7,
      query_processing: 1,
      query_encoding: 0,
      candidate_retrieval: 2,
      fusion: 1,
      reranking: 0,
      materialization: 3,
    },
  };
};
