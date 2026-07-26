import { requestJson } from './client';

// Executes one search turn; session context is optional for non-conversation search.
export const searchFrames = async ({ query, topK, searchMode, sessionId, feedback, signal }) => {
  const body = { query: query.trim(), top_k: topK, search_mode: searchMode };
  if (sessionId) body.session_id = sessionId;
  if (feedback) body.feedback = feedback;

  const payload = await requestJson('/api/v1/search', { method: 'POST', body, signal });
  if (!Array.isArray(payload?.results) || !payload.latency_ms) {
    throw new Error('Search server returned an invalid response contract');
  }
  return payload;
};