import { requestFormData, requestJson } from './client';

export const roundLatencyMs = (val) => {
  if (typeof val !== 'number' || !Number.isFinite(val)) return val;
  return Math.round(val * 100) / 100;
};

export const normalizeSearchLatency = (latency) => {
  if (!latency || typeof latency !== 'object' || Array.isArray(latency)) {
    return typeof latency === 'number' ? roundLatencyMs(latency) : latency;
  }
  const normalized = { ...latency };
  Object.keys(normalized).forEach((key) => {
    if (typeof normalized[key] === 'number') {
      normalized[key] = roundLatencyMs(normalized[key]);
    }
  });
  return normalized;
};

const hasSearchLatency = (latency) => (
  latency
  && typeof latency === 'object'
  && typeof latency.total_ms === 'number'
);

export const searchFramesByImage = async ({
  imageFile,
  topK = 20,
  signal,
  userId,
}) => {
  if (!imageFile) {
    throw new Error('An image file is required for image search');
  }

  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('top_k', String(topK));

  const payload = await requestFormData('/api/v1/search/image', formData, {
    signal,
    headers: userId?.trim() ? { 'X-VBS-User-ID': userId.trim() } : {},
  });

  if (
    !Array.isArray(payload?.results)
    || !hasSearchLatency(payload?.latency)
  ) {
    throw new Error('Image search server returned an invalid response contract');
  }

  return {
    ...payload,
    latency: normalizeSearchLatency(payload.latency),
  };
};
