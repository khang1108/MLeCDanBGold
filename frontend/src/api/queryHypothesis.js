import { requestJson } from './client';

export const openQueryHypothesis = ({ text, imageRefs = [], signal } = {}) => requestJson(
  '/api/v1/kis/hypotheses/open',
  { method: 'POST', body: { text, image_refs: imageRefs }, signal },
);

export const getQueryHypothesis = (sessionId, { signal } = {}) => requestJson(
  `/api/v1/kis/hypotheses/${encodeURIComponent(sessionId)}`,
  { method: 'GET', signal },
);

export const previewQueryHypothesis = (
  sessionId,
  expectedQueryRevision,
  action,
  { signal } = {},
) => requestJson(
  `/api/v1/kis/hypotheses/${encodeURIComponent(sessionId)}/preview`,
  {
    method: 'POST',
    body: { expected_query_revision: expectedQueryRevision, action },
    signal,
  },
);

export const commitQueryHypothesis = (
  sessionId,
  expectedQueryRevision,
  action,
  { signal } = {},
) => requestJson(
  `/api/v1/kis/hypotheses/${encodeURIComponent(sessionId)}/commit`,
  {
    method: 'POST',
    body: { expected_query_revision: expectedQueryRevision, action },
    signal,
  },
);

export const undoQueryHypothesis = (
  sessionId,
  expectedQueryRevision,
  { signal } = {},
) => requestJson(
  `/api/v1/kis/hypotheses/${encodeURIComponent(sessionId)}/undo`,
  {
    method: 'POST',
    body: { expected_query_revision: expectedQueryRevision },
    signal,
  },
);
