import { requestJson } from './client';

/** Validate and reduce a VBS response to the public, browser-safe session state. */
const safeSessionStatus = (payload, expectedUserId) => {
  if (
    !payload
    || typeof payload.user_id !== 'string'
    || payload.user_id.trim() !== expectedUserId
    || typeof payload.connected !== 'boolean'
  ) {
    throw new Error('VBS session server returned an invalid response contract');
  }
  return { user_id: expectedUserId, connected: payload.connected };
};

const normalizeUserId = (value) => {
  const userId = typeof value === 'string' ? value.trim() : '';
  if (!userId) throw new Error('A VBS User ID is required');
  return userId;
};

/** Establish the backend-mapped DRES session using only a participant ID. */
export const connectVbsSession = async (value) => {
  const userId = normalizeUserId(value);
  const payload = await requestJson('/api/v1/vbs/session/connect', {
    method: 'POST',
    body: { user_id: userId },
  });
  return safeSessionStatus(payload, userId);
};

/** Check whether the backend still holds this participant's private session. */
export const getVbsSessionStatus = async (value) => {
  const userId = normalizeUserId(value);
  const payload = await requestJson(`/api/v1/vbs/session/${encodeURIComponent(userId)}`);
  return safeSessionStatus(payload, userId);
};

/** Evict this participant's backend session without exposing its credentials. */
export const disconnectVbsSession = async (value) => {
  const userId = normalizeUserId(value);
  const payload = await requestJson(`/api/v1/vbs/session/${encodeURIComponent(userId)}`, {
    method: 'DELETE',
  });
  return safeSessionStatus(payload, userId);
};

/** Fetch evaluations and their task templates for the connected participant. */
export const getVbsEvaluations = async (value, { signal } = {}) => {
  const userId = normalizeUserId(value);
  const payload = await requestJson(`/api/v1/vbs/evaluations/${encodeURIComponent(userId)}`, { signal });
  if (!Array.isArray(payload)) {
    throw new Error('VBS evaluation list returned an invalid response contract');
  }
  return payload.map((item) => ({
    id: String(item.id || ''),
    name: String(item.name || ''),
    type: String(item.type || ''),
    status: String(item.status || ''),
    templateId: String(item.template_id || ''),
    taskTemplates: Array.isArray(item.task_templates)
      ? item.task_templates.map((t) => ({
          name: String(t.name || ''),
          taskGroup: String(t.task_group || ''),
          taskType: String(t.task_type || ''),
          duration: typeof t.duration === 'number' ? t.duration : null,
        }))
      : [],
  }));
};

