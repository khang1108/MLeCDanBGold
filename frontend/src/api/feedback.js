/**
 * Thin HTTP client for KIS chat feedback, turn execution, and Undo.
 */

import { requestJson } from './client';

export const openFeedbackSession = async ({
  intent,
  originalQuery,
  evidenceSnapshotId,
  useDense = true,
  useBm25 = true,
  topK = 20,
  initialResults = [],
  signal,
}) => {
  if (!intent || typeof intent !== 'object') {
    throw new Error('Intent is required to open a feedback session');
  }
  if (!originalQuery || typeof originalQuery !== 'string') {
    throw new Error('Original query is required to open a feedback session');
  }
  if (!evidenceSnapshotId || typeof evidenceSnapshotId !== 'string') {
    throw new Error('Evidence snapshot ID is required to open a feedback session');
  }

  const body = {
    intent,
    original_query: originalQuery.trim(),
    evidence_snapshot_id: evidenceSnapshotId,
    use_dense: useDense,
    use_bm25: useBm25,
    top_k: topK,
    initial_results: Array.isArray(initialResults) ? initialResults : [],
  };

  const data = await requestJson('/api/v1/kis/feedback/open', {
    method: 'POST',
    body,
    signal,
  });

  if (!data?.session_id || typeof data?.feedback_revision !== 'number' || !data?.state) {
    throw new Error('Malformed feedback open response');
  }
  return data;
};

export const sendFeedbackTurn = async (
  sessionId,
  {
    requestId,
    expectedFeedbackRevision,
    expectedKisRevision,
    expectedTrailRevision,
    message,
    selectedResultId,
    selectedEventId,
    selectedFrameId,
  },
  { signal } = {},
) => {
  if (!sessionId) {
    throw new Error('Session ID is required to send feedback turn');
  }
  if (!requestId) {
    throw new Error('Request ID is required to send feedback turn');
  }
  if (!message || !message.trim()) {
    throw new Error('Message is required to send feedback turn');
  }

  const body = {
    request_id: requestId,
    expected_feedback_revision: expectedFeedbackRevision,
    expected_kis_revision: expectedKisRevision,
    message: message.trim(),
  };

  if (expectedTrailRevision !== undefined && expectedTrailRevision !== null) {
    body.expected_trail_revision = expectedTrailRevision;
  }
  if (selectedResultId) body.selected_result_id = selectedResultId;
  if (selectedEventId) body.selected_event_id = selectedEventId;
  if (selectedFrameId) body.selected_frame_id = selectedFrameId;

  const data = await requestJson(`/api/v1/kis/feedback/${encodeURIComponent(sessionId)}/turn`, {
    method: 'POST',
    body,
    signal,
  });

  if (!data?.session_id || typeof data?.feedback_revision !== 'number') {
    throw new Error('Malformed feedback turn response');
  }
  return data;
};

export const undoFeedback = async (
  sessionId,
  { requestId, expectedFeedbackRevision },
  { signal } = {},
) => {
  if (!sessionId) {
    throw new Error('Session ID is required to undo feedback');
  }
  if (!requestId) {
    throw new Error('Request ID is required to undo feedback');
  }

  const body = {
    request_id: requestId,
    expected_feedback_revision: expectedFeedbackRevision,
  };

  const data = await requestJson(`/api/v1/kis/feedback/${encodeURIComponent(sessionId)}/undo`, {
    method: 'POST',
    body,
    signal,
  });

  if (!data?.session_id || typeof data?.feedback_revision !== 'number') {
    throw new Error('Malformed feedback undo response');
  }
  return data;
};
