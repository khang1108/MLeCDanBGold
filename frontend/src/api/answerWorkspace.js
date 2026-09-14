/** HTTP and WebSocket transport contracts for the shared structured answer workspace. */
import { API_BASE_URL, requestJson } from './client';

const requireText = (value, field) => {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${field} must be a non-blank string`);
  return value;
};

const requireTimestamp = (value, field) => {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${field} must be a non-negative integer`);
  return value;
};

/** Normalize one FRAME or TEXT row and reject mixed answer shapes. */
export const normalizeAnswerCandidate = (candidate, index = 0) => {
  const field = `Answer candidate ${index}`;
  if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
    throw new Error(`${field} must be an object`);
  }
  requireText(candidate.candidate_id, `${field}.candidate_id`);
  requireText(candidate.contributed_by_user_id, `${field}.contributed_by_user_id`);
  requireText(candidate.evaluation_id, `${field}.evaluation_id`);
  requireText(candidate.task_scope_key, `${field}.task_scope_key`);
  requireTimestamp(candidate.created_at_ms, `${field}.created_at_ms`);
  if (!Number.isSafeInteger(candidate.revision) || candidate.revision < 1) {
    throw new Error(`${field}.revision must be a positive integer`);
  }

  if (candidate.kind === 'FRAME') {
    if (candidate.text != null) throw new Error('FRAME candidates cannot contain text');
    requireText(candidate.video_id, `${field}.video_id`);
    requireTimestamp(candidate.timestamp_ms, `${field}.timestamp_ms`);
    if (candidate.source_frame_id != null) requireText(candidate.source_frame_id, `${field}.source_frame_id`);
  } else if (candidate.kind === 'TEXT') {
    if (candidate.video_id != null || candidate.timestamp_ms != null || candidate.source_frame_id != null) {
      throw new Error('TEXT candidates cannot contain frame fields');
    }
    requireText(candidate.text, `${field}.text`);
  } else {
    throw new Error(`${field}.kind must be FRAME or TEXT`);
  }

  return {
    candidate_id: candidate.candidate_id,
    kind: candidate.kind,
    source_frame_id: candidate.source_frame_id ?? null,
    video_id: candidate.video_id ?? null,
    timestamp_ms: candidate.timestamp_ms ?? null,
    text: candidate.text ?? null,
    contributed_by_user_id: candidate.contributed_by_user_id,
    created_at_ms: candidate.created_at_ms,
    revision: candidate.revision,
    submitted_at_ms: candidate.submitted_at_ms ?? null,
    submitted_by_user_id: candidate.submitted_by_user_id ?? null,
    dres_status: candidate.dres_status ?? null,
    evaluation_id: candidate.evaluation_id,
    task_scope_key: candidate.task_scope_key,
  };
};

/** Validate the task scope and normalize every typed candidate in a snapshot. */
export const normalizeAnswerWorkspace = (workspace) => {
  if (!workspace || typeof workspace !== 'object' || Array.isArray(workspace)) {
    throw new Error('Answer workspace server returned an invalid workspace object');
  }
  if (typeof workspace.avs_enabled !== 'boolean') {
    throw new Error('Answer workspace avs_enabled must be a boolean');
  }
  if (!Number.isSafeInteger(workspace.revision) || workspace.revision < 0) {
    throw new Error('Answer workspace revision must be a non-negative integer');
  }
  if (typeof workspace.updated_by_user_id !== 'string') {
    throw new Error('Answer workspace updated_by_user_id must be a string');
  }
  requireTimestamp(workspace.updated_at_ms, 'Answer workspace updated_at_ms');
  requireText(workspace.active_evaluation_id, 'Answer workspace active_evaluation_id');
  requireText(workspace.active_task_scope_key, 'Answer workspace active_task_scope_key');
  requireText(workspace.active_task_name, 'Answer workspace active_task_name');
  if (!Object.hasOwn(workspace, 'task_scope_key') || !Object.hasOwn(workspace, 'task_name')) {
    throw new Error('Answer workspace must include task_scope_key and task_name');
  }
  const taskScopeKey = workspace.task_scope_key ?? null;
  const taskName = workspace.task_name ?? null;
  if (taskScopeKey === null) {
    if (taskName !== null) throw new Error('Answer workspace task_name requires a task_scope_key');
  } else {
    requireText(taskScopeKey, 'Answer workspace task_scope_key');
    requireText(taskName, 'Answer workspace task_name');
  }
  if (!Array.isArray(workspace.candidates)) {
    throw new Error('Answer workspace candidates must be an array');
  }
  if (workspace.task_scope_mismatch !== undefined && typeof workspace.task_scope_mismatch !== 'boolean') {
    throw new Error('Answer workspace task_scope_mismatch must be a boolean');
  }

  const candidates = workspace.candidates.map(normalizeAnswerCandidate);
  return {
    avs_enabled: workspace.avs_enabled,
    evaluation_id: workspace.evaluation_id ?? null,
    task_scope_key: taskScopeKey,
    task_name: taskName,
    revision: workspace.revision,
    updated_by_user_id: workspace.updated_by_user_id,
    updated_at_ms: workspace.updated_at_ms,
    candidates,
    pending_submission: workspace.pending_submission ?? null,
    active_evaluation_id: workspace.active_evaluation_id,
    active_task_scope_key: workspace.active_task_scope_key,
    active_task_name: workspace.active_task_name,
    task_scope_mismatch: workspace.task_scope_mismatch ?? false,
  };
};

/** Fetch the current workspace using only the connected participant identity. */
export const getAnswerWorkspace = async ({ userId, signal } = {}) => {
  const connectedUserId = requireText(userId, 'A connected VBS user ID');
  const payload = await requestJson('/api/v1/answer-workspace', {
    signal,
    headers: { 'X-VBS-User-ID': connectedUserId.trim() },
  });
  return normalizeAnswerWorkspace(payload);
};

/** Forward one frozen KIS FRAME or VQA TEXT candidate through the HCMAI API. */
export const submitAnswerCandidate = async ({
  kind, userId, taskScopeKey, expectedWorkspaceRevision, candidateId, expectedCandidateRevision,
} = {}) => {
  const route = kind === 'FRAME' ? 'kis' : kind === 'TEXT' ? 'vqa' : null;
  if (!route) throw new Error('A submission candidate kind must be FRAME or TEXT');
  return requestJson(`/api/v1/vbs/submit/${route}`, {
    method: 'POST',
    body: {
      user_id: requireText(userId, 'A connected VBS user ID').trim(),
      task_scope_key: requireText(taskScopeKey, 'taskScopeKey'),
      expected_workspace_revision: expectedWorkspaceRevision,
      candidate_id: requireText(candidateId, 'candidateId'),
      expected_revision: expectedCandidateRevision,
    },
  });
};

/** Forward the complete ordered AVS frame sequence in one HCMAI API request. */
export const submitAvsAnswers = async ({
  userId, taskScopeKey, expectedWorkspaceRevision, candidates,
} = {}) => {
  if (!Array.isArray(candidates) || candidates.length === 0) {
    throw new Error('AVS submission requires at least one candidate');
  }
  return requestJson('/api/v1/vbs/submit/avs', {
    method: 'POST',
    body: {
      user_id: requireText(userId, 'A connected VBS user ID').trim(),
      task_scope_key: requireText(taskScopeKey, 'taskScopeKey'),
      expected_workspace_revision: expectedWorkspaceRevision,
      candidates: candidates.map((candidate, index) => ({
        candidate_id: requireText(candidate.candidateId, `candidates[${index}].candidateId`),
        expected_revision: candidate.expectedRevision,
      })),
    },
  });
};

/** Resolve one UNKNOWN DRES attempt through the backend without resubmitting it. */
export const resolveSubmissionAttempt = async ({ userId, attemptId, outcome, signal } = {}) => {
  if (outcome !== 'accepted' && outcome !== 'not_accepted') {
    throw new Error('Attempt outcome must be accepted or not_accepted');
  }
  return requestJson(`/api/v1/vbs/submission-attempts/${encodeURIComponent(requireText(attemptId, 'attemptId'))}/resolve`, {
    method: 'POST',
    signal,
    body: {
      user_id: requireText(userId, 'A connected VBS user ID').trim(),
      outcome,
    },
  });
};

/** Build the browser-safe answer-workspace WebSocket URL for one connected ID. */
export const answerWorkspaceWebSocketUrl = (value) => {
  const connectedUserId = requireText(value, 'A connected VBS user ID');
  const url = new URL('/api/v1/answer-workspace/ws', API_BASE_URL);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.search = `?user_id=${encodeURIComponent(connectedUserId.trim())}`;
  return url.toString();
};
