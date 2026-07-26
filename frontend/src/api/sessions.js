import { requestJson } from './client';

// Session endpoints own server-side turns and committed feedback.
export const createSession = (problemId, signal) => {
  const query = problemId ? `?problem_id=${encodeURIComponent(problemId)}` : '';
  return requestJson(`/api/v1/session${query}`, { method: 'POST', signal });
};

export const listSessions = (signal) => requestJson('/api/v1/sessions', { signal });

export const getSession = (sessionId, signal) => (
  requestJson(`/api/v1/session/${encodeURIComponent(sessionId)}`, { signal })
);

export const updateFeedback = (sessionId, feedback, signal) => (
  requestJson(`/api/v1/feedback?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'POST', body: feedback, signal,
  })
);