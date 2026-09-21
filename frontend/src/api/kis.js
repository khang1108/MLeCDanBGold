import { API_BASE_URL, requestJson, requestFormData } from './client';
import { normalizeSearchLatency } from './search';

export const kisImageAssetUrl = (assetId) => (
  `${API_BASE_URL}/api/v1/kis/assets/images/${encodeURIComponent(assetId)}`
);

export const uploadKisImage = async ({ imageFile, signal }) => {
  const body = new FormData();
  body.append('file', imageFile);
  return requestFormData('/api/v1/kis/assets/images', body, { method: 'POST', signal });
};

const hasSearchLatency = (latency) => (
  latency
  && typeof latency === 'object'
  && typeof latency.total_ms === 'number'
);

export const searchKis = async ({
  queryHypothesisSessionId = null,
  baseIntent = null,
  expectedRevision = 0,
  operation,
  useDense = true,
  useBm25 = true,
  topK = 20,
  userId,
  signal,
}) => {
  if (!useDense && !useBm25) {
    throw new Error('Enable at least one retrieval source');
  }
  if (!operation || typeof operation !== 'object') {
    throw new Error('Operation is required');
  }

  const payload = await requestJson('/api/v1/kis/search', {
    method: 'POST',
    body: {
      query_hypothesis_session_id: queryHypothesisSessionId ?? null,
      base_intent: baseIntent ?? null,
      expected_revision: expectedRevision,
      operation,
      use_dense: useDense,
      use_bm25: useBm25,
      top_k: topK,
    },
    signal,
    headers: userId?.trim() ? { 'X-VBS-User-ID': userId.trim() } : {},
  });

  if (
    !payload?.intent
    || !payload?.operation_summary
    || !Array.isArray(payload?.results)
    || !hasSearchLatency(payload?.latency)
  ) {
    throw new Error('Search server returned an invalid response contract');
  }

  return {
    ...payload,
    latency: normalizeSearchLatency(payload.latency),
  };
};
