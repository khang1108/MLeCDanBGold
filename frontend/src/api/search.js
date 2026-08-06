import { API_BASE_URL, requestJson } from './client';

export const resolveApiUrl = (value) => {
  if (!value || /^(?:https?:|data:)/i.test(value)) return value;
  return `${API_BASE_URL}/${value.replace(/^\/+/, '')}`;
};

const withAssetUrls = (payload) => ({
  ...payload,
  results: payload.results.map((frame) => ({
    ...frame,
    thumbnail_url: resolveApiUrl(frame.thumbnail_url),
    frame_url: resolveApiUrl(frame.frame_url),
  })),
});

// Executes one search turn; session context is optional for non-conversation search.
export const searchFrames = async ({
  query,
  topK,
  sessionId,
  feedback,
  signal,
}) => {
  const body = {
    query: query.trim(),
    top_k: topK,
  };
  if (sessionId) body.session_id = sessionId;
  if (feedback) body.feedback = feedback;

  const payload = await requestJson('/api/v1/search', {
    method: 'POST',
    body,
    signal,
  });
  if (!Array.isArray(payload?.results) || !payload.latency_ms) {
    throw new Error('Search server returned an invalid response contract');
  }
  return withAssetUrls(payload);
};

export const searchVqa = async ({
  eventDescription,
  question,
  topK,
  signal,
}) => {
  const payload = await requestJson('/api/v1/vqa', {
    method: 'POST',
    body: {
      query_type: 'vqa',
      event_description: eventDescription.trim(),
      question: question.trim(),
      top_k: topK,
    },
    signal,
  });
  if (!Array.isArray(payload?.submissions)
      || typeof payload?.latency_ms !== 'number') {
    throw new Error('VQA server returned an invalid response contract');
  }
  return payload;
};

export const searchKisc = async ({
  history = [],
  currentMessage,
  previousState = null,
  feedback = {},
  topK = 20,
  filters = null,
  signal,
}) => {
  const payload = await requestJson('/api/v1/kisc/search', {
    method: 'POST',
    body: {
      history,
      current_message: currentMessage.trim(),
      previous_state: previousState,
      feedback,
      top_k: topK,
      filters,
    },
    signal,
  });
  if (!payload?.interpreted_state || !Array.isArray(payload?.search?.results)) {
    throw new Error('KISC server returned an invalid response contract');
  }
  return { ...payload, search: withAssetUrls(payload.search) };
};
