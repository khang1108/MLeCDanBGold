const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

export const API_BASE_URL = (
  process.env.REACT_APP_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/+$/, '');

const errorMessage = (payload, status) => {
  if (typeof payload?.detail === 'string') return payload.detail;
  if (typeof payload?.detail?.message === 'string') return payload.detail.message;
  return `Search request failed with HTTP ${status}`;
};

export const searchFrames = async ({
  query,
  topK,
  searchMode,
  signal,
}) => {
  const response = await fetch(`${API_BASE_URL}/api/v1/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: query.trim(),
      top_k: topK,
      search_mode: searchMode,
    }),
    signal,
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(errorMessage(payload, response.status));
  }
  if (!payload || !Array.isArray(payload.results) || !payload.latency_ms) {
    throw new Error('Search server returned an invalid response contract');
  }
  return payload;
};
