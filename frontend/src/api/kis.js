import { requestJson } from './client';
import { normalizeSearchLatency } from './search';

const hasSearchLatency = (latency) => (
  latency
  && typeof latency === 'object'
  && typeof latency.total_ms === 'number'
);

export const searchKis = async ({
  inputs,
  expectedRevision = 0,
  useDense = true,
  useBm25 = true,
  topK = 20,
  userId,
  signal,
}) => {
  if (!useDense && !useBm25) {
    throw new Error('Enable at least one retrieval source');
  }
  if (!Array.isArray(inputs) || inputs.length === 0) {
    throw new Error('Inputs must be a non-empty array');
  }

  const normalizedInputs = inputs.map((item) => {
    const text = typeof item === 'string' ? item.trim() : item?.text?.trim();
    if (!text) {
      throw new Error('Input text must not be blank');
    }
    return { text };
  });

  const payload = await requestJson('/api/v1/kis/search', {
    method: 'POST',
    body: {
      inputs: normalizedInputs,
      expected_revision: expectedRevision,
      use_dense: useDense,
      use_bm25: useBm25,
      top_k: topK,
    },
    signal,
    headers: userId?.trim() ? { 'X-VBS-User-ID': userId.trim() } : {},
  });

  if (
    !payload?.intent
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
