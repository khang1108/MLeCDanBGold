import { requestJson } from './client';

const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const nonBlank = (value) => typeof value === 'string' && Boolean(value.trim());

const validateCandidate = (item) => {
  if (!isRecord(item)) throw new Error('Invalid AVS candidate item');
  if (!nonBlank(item.candidate_id) || !nonBlank(item.frame_id)) {
    throw new Error('Candidate identity must be non-empty');
  }
  if (item.candidate_id !== item.frame_id) {
    throw new Error('candidate_id must match frame_id');
  }
  if (!nonBlank(item.video_id)) throw new Error('video_id must be non-empty');
  if (!Number.isSafeInteger(item.timestamp_ms) || item.timestamp_ms < 0) {
    throw new Error('timestamp_ms must be non-negative integer');
  }
  if (item.frame_idx !== undefined && (!Number.isSafeInteger(item.frame_idx) || item.frame_idx < 0)) {
    throw new Error('frame_idx must be non-negative integer');
  }
  if (!Number.isSafeInteger(item.retrieval_rank) || item.retrieval_rank < 1) {
    throw new Error('retrieval_rank must be >= 1');
  }
};

const validateResponse = (payload) => {
  if (!isRecord(payload)) throw new Error('Invalid AVS response');
  if (!Array.isArray(payload.results)) throw new Error('results must be an array');
  payload.results.forEach(validateCandidate);

  if (
    !isRecord(payload.latency)
    || typeof payload.latency.total_ms !== 'number'
    || typeof payload.latency.retrieval_ms !== 'number'
  ) {
    throw new Error('Invalid latency object');
  }

  if (
    !Number.isSafeInteger(payload.candidate_pool_size)
    || !Number.isSafeInteger(payload.deduplicated_candidate_count)
    || !Number.isSafeInteger(payload.unique_videos)
    || !Array.isArray(payload.warnings)
  ) {
    throw new Error('Invalid AVS counter or warnings metadata');
  }
  return payload;
};

export const searchAvs = async ({ query, pageSize = 80, userId, signal } = {}) => {
  const normalizedQuery = typeof query === 'string' ? query.trim() : '';
  if (!normalizedQuery) throw new Error('An AVS text query is required');
  if (!Number.isSafeInteger(pageSize) || pageSize < 1) {
    throw new Error('AVS page size must be positive');
  }

  const headers = userId && typeof userId === 'string' && userId.trim()
    ? { 'X-VBS-User-ID': userId.trim() }
    : {};

  const payload = await requestJson('/api/v1/avs/search', {
    method: 'POST',
    body: { query: normalizedQuery, page_size: pageSize },
    signal,
    headers,
  });

  return validateResponse(payload);
};
