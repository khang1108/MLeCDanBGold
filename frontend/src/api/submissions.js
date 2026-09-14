import { requestJson } from './client';

const VERDICTS = new Set(['CORRECT', 'WRONG', 'INDETERMINATE', 'UNDECIDABLE']);
const REASONS = new Set(['DRES_REJECTED', 'DRES_AUTH_REJECTED']);

const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const nonBlank = (value) => typeof value === 'string' && Boolean(value.trim());

const normalizeUserId = (value) => {
  const userId = typeof value === 'string' ? value.trim() : '';
  if (!userId) throw new Error('A VBS User ID is required');
  return userId;
};

const safeTask = (payload, expectedUserId) => {
  if (
    !isRecord(payload)
    || payload.user_id !== expectedUserId
    || !nonBlank(payload.evaluation_id)
    || !nonBlank(payload.task_scope_key)
    || !nonBlank(payload.task_name)
    || !nonBlank(payload.task_group)
    || !nonBlank(payload.task_type)
    || !(payload.duration === null || (Number.isSafeInteger(payload.duration) && payload.duration >= 0))
  ) {
    throw new Error('DRES task lookup returned an invalid response contract');
  }

  return {
    user_id: expectedUserId,
    evaluation_id: payload.evaluation_id,
    task_scope_key: payload.task_scope_key,
    task_name: payload.task_name,
    task_group: payload.task_group,
    task_type: payload.task_type,
    duration: payload.duration,
  };
};

const safeAnswer = (answer) => {
  if (!isRecord(answer)) throw new Error('Exactly one answer object is required');

  if (answer.kind === 'TEMPORAL') {
    if (
      !nonBlank(answer.video_id)
      || !Number.isSafeInteger(answer.start_ms)
      || answer.start_ms < 0
      || !Number.isSafeInteger(answer.end_ms)
      || answer.end_ms < answer.start_ms
    ) {
      throw new Error('Invalid answer: expected one valid temporal range');
    }
    return {
      kind: 'TEMPORAL',
      video_id: answer.video_id,
      start_ms: answer.start_ms,
      end_ms: answer.end_ms,
    };
  }

  if (answer.kind === 'TEXT' && nonBlank(answer.text)) {
    return { kind: 'TEXT', text: answer.text };
  }
  throw new Error('Invalid answer: expected one temporal or text answer');
};

const safeOutcome = (payload) => {
  if (!isRecord(payload) || !nonBlank(payload.message)) {
    throw new Error('DRES returned an invalid submission outcome');
  }

  if (
    payload.state === 'RECORDED'
    && payload.recorded === true
    && VERDICTS.has(payload.verdict)
    && (payload.reason === undefined || payload.reason === null)
  ) {
    return {
      state: 'RECORDED',
      recorded: true,
      verdict: payload.verdict,
      message: payload.message,
    };
  }

  if (
    payload.state === 'NOT_RECORDED'
    && payload.recorded === false
    && (payload.verdict === undefined || payload.verdict === null)
    && REASONS.has(payload.reason)
  ) {
    return {
      state: 'NOT_RECORDED',
      recorded: false,
      reason: payload.reason,
      message: payload.message,
    };
  }

  if (
    payload.state === 'UNKNOWN'
    && payload.recorded === null
    && (payload.verdict === undefined || payload.verdict === null)
    && (payload.reason === undefined || payload.reason === null)
  ) {
    return { state: 'UNKNOWN', recorded: null, message: payload.message };
  }

  throw new Error('DRES returned an invalid submission outcome');
};

/** Resolve the selected participant's secret-free live DRES task metadata. */
export const getCurrentDresTask = async (value, { signal } = {}) => {
  const userId = normalizeUserId(value);
  const payload = await requestJson(`/api/v1/vbs/task/${encodeURIComponent(userId)}`, { signal });
  return safeTask(payload, userId);
};

/** Submit exactly one answer against the popup's frozen DRES task scope. */
export const submitDresAnswer = async ({ userId: value, expectedTaskScopeKey, answer, signal } = {}) => {
  const userId = normalizeUserId(value);
  if (!nonBlank(expectedTaskScopeKey)) throw new Error('A DRES task scope key is required');

  const payload = await requestJson('/api/v1/vbs/submit', {
    method: 'POST',
    body: {
      user_id: userId,
      expected_task_scope_key: expectedTaskScopeKey,
      answer: safeAnswer(answer),
    },
    signal,
  });
  return safeOutcome(payload);
};
