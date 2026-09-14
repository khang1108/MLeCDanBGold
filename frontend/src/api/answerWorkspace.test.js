import {
  answerWorkspaceWebSocketUrl,
  getAnswerWorkspace,
  resolveSubmissionAttempt,
  submitAnswerCandidate,
  submitAvsAnswers,
} from './answerWorkspace';

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

const emptyWorkspace = () => ({
  avs_enabled: false,
  evaluation_id: 'eval-1',
  task_scope_key: 'dres-task-v1:key-1',
  task_name: 'KIS task',
  revision: 3,
  updated_by_user_id: 'team-a',
  updated_at_ms: 100,
  candidates: [],
  pending_submission: null,
  active_evaluation_id: 'eval-1',
  active_task_scope_key: 'dres-task-v1:key-1',
  active_task_name: 'KIS task',
  task_scope_mismatch: false,
});

afterEach(() => jest.restoreAllMocks());

test('hydrates the answer workspace with only the connected-user header', async () => {
  const payload = emptyWorkspace();
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  await expect(getAnswerWorkspace({ userId: 'team-a' })).resolves.toEqual(payload);
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/answer-workspace'),
    expect.objectContaining({
      method: 'GET',
      headers: expect.objectContaining({ 'X-VBS-User-ID': 'team-a' }),
    }),
  );
  expect(JSON.stringify(global.fetch.mock.calls)).not.toMatch(/credential|password|dres_session|session_id/i);
});

test('accepts a workspace with no bound task while retaining the active task name', async () => {
  const payload = {
    ...emptyWorkspace(),
    evaluation_id: null,
    task_scope_key: null,
    task_name: null,
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  await expect(getAnswerWorkspace({ userId: 'team-a' })).resolves.toMatchObject({
    task_scope_key: null,
    task_name: null,
    active_task_scope_key: 'dres-task-v1:key-1',
    active_task_name: 'KIS task',
  });
});

test('requires the renamed scope fields instead of hydrating the retired task ID contract', async () => {
  const payload = emptyWorkspace();
  delete payload.task_scope_key;
  delete payload.task_name;
  payload[['task', 'id'].join('_')] = 'old-task-id';
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  await expect(getAnswerWorkspace({ userId: 'team-a' }))
    .rejects.toThrow(/must include task_scope_key and task_name/i);
});

test('does not request shared state without a connected user ID', async () => {
  const fetchSpy = jest.spyOn(global, 'fetch');

  await expect(getAnswerWorkspace({ userId: '' })).rejects.toThrow(/connected VBS user/i);
  expect(fetchSpy).not.toHaveBeenCalled();
});

test('builds a browser WebSocket URL with an encoded connected user ID', () => {
  const url = answerWorkspaceWebSocketUrl('team /a');
  expect(url).toContain('/api/v1/answer-workspace/ws?user_id=team%20%2Fa');
  expect(url).toMatch(/^wss?:\/\//);
  expect(url).not.toMatch(/dres|session|credential|password/i);
});

test('rejects malformed mixed FRAME and TEXT candidates during hydration', async () => {
  const candidate = {
    candidate_id: 'candidate-1',
    kind: 'FRAME',
    source_frame_id: null,
    video_id: 'V01',
    timestamp_ms: 1000,
    text: 'mixed content',
    contributed_by_user_id: 'team-a',
    created_at_ms: 1,
    revision: 1,
    submitted_at_ms: null,
    submitted_by_user_id: null,
    dres_status: null,
    evaluation_id: 'eval-1',
    task_scope_key: 'dres-task-v1:key-1',
  };
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    ...emptyWorkspace(), candidates: [candidate],
  }));

  await expect(getAnswerWorkspace({ userId: 'team-a' }))
    .rejects.toThrow(/FRAME candidates cannot contain text/i);
});

test('submits one frozen FRAME candidate through the KIS backend route', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({ accepted: true }));

  await submitAnswerCandidate({
    kind: 'FRAME',
    userId: 'team-a',
    taskScopeKey: 'dres-task-v1:kis',
    expectedWorkspaceRevision: 7,
    candidateId: 'frame-1',
    expectedCandidateRevision: 3,
  });

  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/vbs/submit/kis'),
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        user_id: 'team-a',
        task_scope_key: 'dres-task-v1:kis',
        expected_workspace_revision: 7,
        candidate_id: 'frame-1',
        expected_revision: 3,
      }),
    }),
  );
  expect(JSON.stringify(global.fetch.mock.calls)).not.toMatch(/credential|password|dres_session|session_id/i);
});

test('submits one frozen TEXT candidate through the VQA backend route', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({ accepted: true }));

  await submitAnswerCandidate({
    kind: 'TEXT',
    userId: 'team-a',
    taskScopeKey: 'dres-task-v1:vqa',
    expectedWorkspaceRevision: 8,
    candidateId: 'text-1',
    expectedCandidateRevision: 2,
  });

  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/vbs/submit/vqa'),
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        user_id: 'team-a',
        task_scope_key: 'dres-task-v1:vqa',
        expected_workspace_revision: 8,
        candidate_id: 'text-1',
        expected_revision: 2,
      }),
    }),
  );
});

test('forwards the complete AVS candidate order in one backend request', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({ accepted: true }));
  const candidates = [
    { candidateId: 'frame-b', expectedRevision: 4 },
    { candidateId: 'frame-a', expectedRevision: 6 },
  ];

  await submitAvsAnswers({
    userId: 'team-a',
    taskScopeKey: 'dres-task-v1:avs',
    expectedWorkspaceRevision: 11,
    candidates,
  });

  expect(global.fetch).toHaveBeenCalledTimes(1);
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/vbs/submit/avs'),
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        user_id: 'team-a',
        task_scope_key: 'dres-task-v1:avs',
        expected_workspace_revision: 11,
        candidates: [
          { candidate_id: 'frame-b', expected_revision: 4 },
          { candidate_id: 'frame-a', expected_revision: 6 },
        ],
      }),
    }),
  );
});

test('resolves an UNKNOWN attempt only through the backend resolution route', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({ state: 'ACCEPTED' }));

  await resolveSubmissionAttempt({
    userId: 'team-a', attemptId: 'attempt/1', outcome: 'accepted',
  });

  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/vbs/submission-attempts/attempt%2F1/resolve'),
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ user_id: 'team-a', outcome: 'accepted' }),
    }),
  );
});
