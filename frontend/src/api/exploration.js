/** Thin transport for server-owned temporal exploration branches. */
import { requestJson } from './client';

/** Open an exploration branch from an immutable retrieval snapshot. */
export const openExploration = (body, { signal } = {}) => requestJson('/api/v1/exploration', {
  method: 'POST',
  body,
  signal,
});

/** Read the current server view for one exploration branch. */
export const getExploration = (handle, { signal } = {}) => requestJson(
  `/api/v1/exploration/${encodeURIComponent(handle)}`,
  { signal },
);

/** Apply one revision-guarded feedback action to an exploration branch. */
export const actOnExploration = (handle, body, { signal } = {}) => requestJson(
  `/api/v1/exploration/${encodeURIComponent(handle)}/actions`,
  { method: 'POST', body, signal },
);

/** Close one branch after its latest revision has been observed. */
export const closeExploration = (handle, revision, { signal } = {}) => requestJson(
  `/api/v1/exploration/${encodeURIComponent(handle)}?expected_revision=${encodeURIComponent(revision)}`,
  { method: 'DELETE', signal },
);
