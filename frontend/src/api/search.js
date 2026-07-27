const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

export const API_BASE_URL = (
  process.env.REACT_APP_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/+$/, '');

export const resolveApiUrl = (value) => {
  if (!value || /^(?:https?:|data:)/i.test(value)) return value;
  return `${API_BASE_URL}/${value.replace(/^\/+/, '')}`;
};

const errorMessage = (payload, status) => {
  if (typeof payload?.detail === 'string') return payload.detail;
  if (typeof payload?.detail?.message === 'string') return payload.detail.message;
  return `Search request failed with HTTP ${status}`;
};

const postJson = async (path, body, signal) => {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(errorMessage(payload, response.status));
  }
  return payload;
};

const withAssetUrls = (payload) => ({
  ...payload,
  results: payload.results.map((frame) => ({
    ...frame,
    thumbnail_url: resolveApiUrl(frame.thumbnail_url),
    frame_url: resolveApiUrl(frame.frame_url),
  })),
});

export const searchFrames = async ({
  query,
  topK,
  searchMode,
  signal,
}) => {
  const payload = await postJson(
    '/api/v1/search',
    {
      query: query.trim(),
      top_k: topK,
      search_mode: searchMode,
    },
    signal,
  );
  if (!payload || !Array.isArray(payload.results) || !payload.latency_ms) {
    throw new Error('Search server returned an invalid response contract');
  }
  return withAssetUrls(payload);
};

export const searchKisc = async ({
  history = [],
  currentMessage,
  previousState = null,
  feedback = {},
  topK = 20,
  searchMode = 'accurate',
  filters = null,
  signal,
}) => {
  const payload = await postJson('/api/v1/kisc/search', {
    history,
    current_message: currentMessage.trim(),
    previous_state: previousState,
    feedback,
    top_k: topK,
    search_mode: searchMode,
    filters,
  }, signal);
  if (!payload?.interpreted_state || !Array.isArray(payload?.search?.results)) {
    throw new Error('KISC server returned an invalid response contract');
  }
  return {...payload, search: withAssetUrls(payload.search)};
};

export const answerFrameQuestion = async ({frameId, question, signal}) => {
  const payload = await postJson('/api/v1/vqa', {
    frame_id: frameId,
    question: question.trim(),
  }, signal);
  if (!payload?.frame_id || typeof payload.answer !== 'string') {
    throw new Error('VQA server returned an invalid response contract');
  }
  return payload;
};
