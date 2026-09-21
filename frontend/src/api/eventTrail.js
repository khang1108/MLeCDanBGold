// Thin EventTrail transport client and structural validation.
import { requestJson } from './client';

const validateTrailState = (state) => {
  if (!state || typeof state !== 'object') {
    throw new Error('Malformed EventTrail response: expected state object');
  }
  if (!state.session_id || typeof state.session_id !== 'string') {
    throw new Error('Malformed EventTrail response: missing session_id');
  }
  if (typeof state.trail_revision !== 'number') {
    throw new Error('Malformed EventTrail response: missing trail_revision');
  }
  if (state.status !== 'active' && state.status !== 'exhausted') {
    throw new Error(`Malformed EventTrail response: invalid status "${state.status}"`);
  }
  if (state.path !== null && !Array.isArray(state.path)) {
    throw new Error('Malformed EventTrail response: invalid path shape');
  }
  return state;
};

export const openEventTrail = async ({
  snapshotId,
  resultId,
  expectedKisRevision,
  searchSessionId,
  signal: inputSignal,
}, { signal = inputSignal } = {}) => {
  const body = {
    snapshot_id: snapshotId,
    result_id: resultId,
    expected_kis_revision: expectedKisRevision,
  };
  if (searchSessionId !== undefined && searchSessionId !== null) {
    body.search_session_id = searchSessionId;
  }
  const data = await requestJson('/api/v1/event-trail/open', {
    method: 'POST',
    body,
    signal,
  });
  return validateTrailState(data);
};

export const getEventTrail = async (sessionId, { signal } = {}) => {
  const data = await requestJson(`/api/v1/event-trail/${encodeURIComponent(sessionId)}`, {
    method: 'GET',
    signal,
  });
  return validateTrailState(data);
};

export const getEventTrailAlternatives = async (
  sessionId,
  { eventId, expectedTrailRevision, signal } = {},
) => {
  const query = `event_id=${encodeURIComponent(eventId)}&expected_trail_revision=${encodeURIComponent(expectedTrailRevision)}`;
  return requestJson(
    `/api/v1/event-trail/${encodeURIComponent(sessionId)}/alternatives?${query}`,
    { method: 'GET', signal },
  );
};

export const actOnEventTrail = async (
  sessionId,
  { expectedTrailRevision, action },
  { signal } = {},
) => {
  const body = {
    expected_trail_revision: expectedTrailRevision,
    action,
  };
  const data = await requestJson(`/api/v1/event-trail/${encodeURIComponent(sessionId)}/actions`, {
    method: 'POST',
    body,
    signal,
  });
  return validateTrailState(data);
};

export const closeEventTrail = async (
  sessionId,
  expectedRevisionOrOptions,
  options = {},
) => {
  let revision;
  let signal;
  if (typeof expectedRevisionOrOptions === 'number') {
    revision = expectedRevisionOrOptions;
    signal = options?.signal;
  } else if (expectedRevisionOrOptions && typeof expectedRevisionOrOptions === 'object') {
    revision = expectedRevisionOrOptions.expectedTrailRevision;
    signal = expectedRevisionOrOptions.signal || options?.signal;
  } else {
    signal = options?.signal;
  }
  const query = Number.isInteger(revision)
    ? `?expected_trail_revision=${encodeURIComponent(revision)}`
    : '';
  return requestJson(`/api/v1/event-trail/${encodeURIComponent(sessionId)}${query}`, {
    method: 'DELETE',
    signal,
  });
};
