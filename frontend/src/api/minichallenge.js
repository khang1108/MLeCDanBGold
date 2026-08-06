import { requestJson } from './client';

const sessionOptions = (session, options = {}) => ({
  ...options,
  headers: { ...options.headers, 'X-DRES-Session': session.trim() },
});

export const listMiniChallengeEvaluations = async (session, signal) => {
  const payload = await requestJson(
    '/api/v1/minichallenge/evaluations',
    sessionOptions(session, { signal }),
  );
  if (!Array.isArray(payload)) {
    throw new Error('Mini-challenge server returned an invalid evaluation list');
  }
  return payload;
};

export const getMiniChallengeCurrentTask = async (
  session,
  evaluationId,
  signal,
) => {
  const payload = await requestJson(
    `/api/v1/minichallenge/evaluations/${encodeURIComponent(evaluationId)}/current-task`,
    sessionOptions(session, { signal }),
  );
  if (!payload?.name || !payload?.taskGroup || !payload?.taskType) {
    throw new Error('Mini-challenge server returned an invalid current task');
  }
  return payload;
};

export const submitMiniChallengeFrame = async ({
  session,
  evaluationId,
  frameId,
  taskName,
  text,
  signal,
}) => {
  const payload = await requestJson(
    `/api/v1/minichallenge/evaluations/${encodeURIComponent(evaluationId)}/submit`,
    sessionOptions(session, {
      method: 'POST',
      body: {
        frame_id: frameId,
        task_name: taskName,
        text: text.trim() || null,
      },
      signal,
    }),
  );
  if (typeof payload?.status !== 'boolean' || !payload?.submission) {
    throw new Error('Mini-challenge server returned an invalid submission result');
  }
  return payload;
};
