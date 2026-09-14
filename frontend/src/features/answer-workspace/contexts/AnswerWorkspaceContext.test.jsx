import React from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import AnswerWorkspaceProvider, { useAnswerWorkspace } from './AnswerWorkspaceContext';
import { answerWorkspaceWebSocketUrl, getAnswerWorkspace } from '../../../api/answerWorkspace';

jest.mock('../../../api/answerWorkspace', () => ({
  getAnswerWorkspace: jest.fn(),
  answerWorkspaceWebSocketUrl: jest.fn((userId) => (
    `ws://example.test/api/v1/answer-workspace/ws?user_id=${encodeURIComponent(userId)}`
  )),
  normalizeAnswerWorkspace: jest.requireActual('../../../api/answerWorkspace').normalizeAnswerWorkspace,
}));

class MockWebSocket {
  static instances = [];
  static OPEN = 1;

  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.sent = [];
    MockWebSocket.instances.push(this);
  }

  send(value) { this.sent.push(value); }
  open() { this.readyState = 1; this.onopen?.(); }
  message(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }
  close() { this.readyState = 3; this.onclose?.(); }
}

const workspace = (revision = 1, candidates = []) => ({
  avs_enabled: false,
  evaluation_id: 'eval-1',
  task_scope_key: 'dres-task-v1:key-1',
  task_name: 'KIS task',
  revision,
  updated_by_user_id: 'team-a',
  updated_at_ms: revision,
  candidates,
  pending_submission: null,
  active_evaluation_id: 'eval-1',
  active_task_scope_key: 'dres-task-v1:key-1',
  active_task_name: 'KIS task',
  task_scope_mismatch: false,
});

const frameCandidate = ({ candidateId = 'candidate-1', timestampMs = 1000, revision = 1 } = {}) => ({
  candidate_id: candidateId,
  kind: 'FRAME',
  source_frame_id: null,
  video_id: 'V01',
  timestamp_ms: timestampMs,
  text: null,
  contributed_by_user_id: 'team-a',
  created_at_ms: 1,
  revision,
  submitted_at_ms: null,
  submitted_by_user_id: null,
  dres_status: null,
  evaluation_id: 'eval-1',
  task_scope_key: 'dres-task-v1:key-1',
});

const Probe = () => {
  const context = useAnswerWorkspace();
  window.answerWorkspaceProbe = context;
  return (
    <div>
      <output data-testid="revision">{context.workspace?.revision ?? ''}</output>
      <output data-testid="candidates">{context.workspace?.candidates.map((item) => item.candidate_id).join(',') || ''}</output>
      <output data-testid="connected">{String(context.isConnected)}</output>
      <output data-testid="error">{context.connectionError || ''}</output>
    </div>
  );
};

const renderProvider = (connectedUserId = 'team-a') => render(
  <AnswerWorkspaceProvider connectedUserId={connectedUserId}>
    <Probe />
  </AnswerWorkspaceProvider>,
);

const openSocket = async (index = 0) => {
  await act(async () => {
    MockWebSocket.instances[index].open();
    await Promise.resolve();
  });
};

beforeEach(() => {
  jest.clearAllMocks();
  answerWorkspaceWebSocketUrl.mockImplementation((userId) => (
    `ws://example.test/api/v1/answer-workspace/ws?user_id=${encodeURIComponent(userId)}`
  ));
  MockWebSocket.instances = [];
  window.WebSocket = MockWebSocket;
  getAnswerWorkspace.mockResolvedValue(workspace());
});

afterEach(() => {
  delete window.WebSocket;
  delete window.answerWorkspaceProbe;
  jest.useRealTimers();
});

test('hydrates after socket open and connects with only the encoded participant ID', async () => {
  renderProvider('team /a');

  expect(MockWebSocket.instances).toHaveLength(1);
  expect(answerWorkspaceWebSocketUrl).toHaveBeenCalledWith('team /a');
  expect(answerWorkspaceWebSocketUrl.mock.results[0].value).toContain('user_id=team%20%2Fa');
  expect(MockWebSocket.instances[0].url).toContain('user_id=team%20%2Fa');
  expect(getAnswerWorkspace).not.toHaveBeenCalled();
  await openSocket();

  await waitFor(() => expect(screen.getByTestId('revision').textContent).toBe('1'));
  expect(getAnswerWorkspace).toHaveBeenCalledWith({ userId: 'team /a', signal: expect.any(AbortSignal) });
  expect(screen.getByTestId('connected').textContent).toBe('true');
});

test('does not open a socket or permit mutations without a connected user', async () => {
  renderProvider('');

  expect(MockWebSocket.instances).toHaveLength(0);
  expect(getAnswerWorkspace).not.toHaveBeenCalled();
  await expect(window.answerWorkspaceProbe.addText({ text: 'answer' }))
    .rejects.toThrow(/connect a VBS user/i);
});

test('locks every ordinary workspace mutation while an attempt is unresolved', async () => {
  getAnswerWorkspace.mockResolvedValueOnce({
    ...workspace(1),
    pending_submission: {
      attempt_id: 'attempt-1',
      kind: 'KIS',
      state: 'UNKNOWN',
      submitted_by_user_id: 'team-a',
      created_at_ms: 1,
    },
  });
  renderProvider();
  await openSocket();
  await waitFor(() => expect(screen.getByTestId('revision').textContent).toBe('1'));

  const mutation = window.answerWorkspaceProbe.addText({ text: 'answer' });
  mutation.catch(() => {});
  if (MockWebSocket.instances[0].sent.length) {
    await act(async () => MockWebSocket.instances[0].message({
      type: 'answer.error', message: 'Submission attempt is unresolved',
    }));
  }
  await expect(mutation).rejects.toThrow(/submission attempt is unresolved/i);
  expect(MockWebSocket.instances[0].sent).toHaveLength(0);
});

test('clears and switches to the server-reported task through one revisioned WS command', async () => {
  getAnswerWorkspace.mockResolvedValueOnce({
    ...workspace(4, [frameCandidate()]),
    evaluation_id: 'eval-old',
    task_scope_key: 'dres-task-v1:old',
    task_name: 'Old task',
    active_evaluation_id: 'eval-new',
    active_task_scope_key: 'dres-task-v1:new',
    active_task_name: 'New task',
    task_scope_mismatch: true,
  });
  renderProvider();
  await openSocket();
  await waitFor(() => expect(screen.getByTestId('revision').textContent).toBe('4'));

  let switchTask;
  await act(async () => {
    switchTask = window.answerWorkspaceProbe.clearAndSwitchTask();
    switchTask.catch(() => {});
  });
  expect(JSON.parse(MockWebSocket.instances[0].sent[0])).toEqual({
    type: 'answer.task.clear_and_switch',
    expected_workspace_revision: 4,
    expected_old_evaluation_id: 'eval-old',
    expected_old_task_scope_key: 'dres-task-v1:old',
    target_evaluation_id: 'eval-new',
    target_task_scope_key: 'dres-task-v1:new',
  });

  const switchedWorkspace = {
    ...workspace(1),
    evaluation_id: 'eval-new',
    task_scope_key: 'dres-task-v1:new',
    task_name: 'New task',
    active_evaluation_id: 'eval-new',
    active_task_scope_key: 'dres-task-v1:new',
    active_task_name: 'New task',
    task_scope_mismatch: false,
  };
  await act(async () => MockWebSocket.instances[0].message({
    type: 'answer.task.switched', workspace: switchedWorkspace,
  }));
  await expect(switchTask).resolves.toMatchObject({ task_scope_key: 'dres-task-v1:new', candidates: [] });
  expect(screen.getByTestId('revision').textContent).toBe('1');
});

test('replays events buffered while the HTTP snapshot is loading', async () => {
  let resolveHydration;
  getAnswerWorkspace.mockReturnValueOnce(new Promise((resolve) => { resolveHydration = resolve; }));
  renderProvider();
  await openSocket();

  await act(async () => MockWebSocket.instances[0].message({
    type: 'answer.added',
    workspace: workspace(2, [frameCandidate()]),
  }));
  await act(async () => {
    resolveHydration(workspace(1));
    await Promise.resolve();
  });

  expect(screen.getByTestId('revision').textContent).toBe('2');
  expect(screen.getByTestId('candidates').textContent).toBe('candidate-1');
});

test('settles duplicate frame additions at the same revision only for the matching frame', async () => {
  getAnswerWorkspace.mockResolvedValueOnce(workspace(1, [frameCandidate()]));
  renderProvider();
  await openSocket();
  await waitFor(() => expect(screen.getByTestId('revision').textContent).toBe('1'));

  let add;
  await act(async () => {
    add = window.answerWorkspaceProbe.addFrame({ videoId: 'V01', timestampMs: 1000 });
    add.catch(() => {});
  });

  await act(async () => MockWebSocket.instances[0].message({
    type: 'answer.added', workspace: workspace(1, [frameCandidate({ candidateId: 'other', timestampMs: 2000 })]),
  }));
  expect(window.answerWorkspaceProbe.pendingAction).toBe('answer.add_frame');

  await act(async () => MockWebSocket.instances[0].message({
    type: 'answer.added', workspace: workspace(1, [frameCandidate()]),
  }));

  await expect(add).resolves.toMatchObject({ revision: 1 });
  expect(window.answerWorkspaceProbe.pendingAction).toBeNull();
});

test('ignores out-of-order workspace revisions after hydration', async () => {
  renderProvider();
  await openSocket();
  await waitFor(() => expect(screen.getByTestId('revision').textContent).toBe('1'));

  await act(async () => MockWebSocket.instances[0].message({
    type: 'answer.updated', workspace: workspace(3, [frameCandidate({ timestampMs: 3000, revision: 2 })]),
  }));
  await act(async () => MockWebSocket.instances[0].message({
    type: 'answer.updated', workspace: workspace(2, [frameCandidate({ timestampMs: 2000, revision: 2 })]),
  }));

  expect(screen.getByTestId('revision').textContent).toBe('3');
  expect(window.answerWorkspaceProbe.workspace.candidates[0].timestamp_ms).toBe(3000);
});

test('refreshes on revision conflict and rejects the pending mutation', async () => {
  renderProvider();
  await openSocket();
  await waitFor(() => expect(screen.getByTestId('revision').textContent).toBe('1'));

  let update;
  await act(async () => {
    update = window.answerWorkspaceProbe.updateFrame({
      candidateId: 'candidate-1',
      expectedCandidateRevision: 1,
      videoId: 'V01',
      timestampMs: 1500,
    });
    update.catch(() => {});
  });
  expect(JSON.parse(MockWebSocket.instances[0].sent[0])).toEqual({
    type: 'answer.update_frame',
    candidate_id: 'candidate-1',
    expected_candidate_revision: 1,
    video_id: 'V01',
    timestamp_ms: 1500,
    expected_workspace_revision: 1,
  });
  await act(async () => MockWebSocket.instances[0].message({
    type: 'answer.conflict',
    workspace: workspace(2, [frameCandidate({ timestampMs: 2000, revision: 2 })]),
  }));

  await expect(update).rejects.toMatchObject({ code: 'REVISION_CONFLICT' });
  expect(window.answerWorkspaceProbe.workspace.candidates[0].timestamp_ms).toBe(2000);
  expect(screen.getByTestId('revision').textContent).toBe('2');
});

test('times out an unacknowledged mutation and clears its pending state', async () => {
  renderProvider();
  await openSocket();
  await waitFor(() => expect(screen.getByTestId('revision').textContent).toBe('1'));
  jest.useFakeTimers();

  let add;
  await act(async () => {
    add = window.answerWorkspaceProbe.addText({ text: 'VQA answer' });
    add.catch(() => {});
  });
  expect(JSON.parse(MockWebSocket.instances[0].sent[0])).toMatchObject({
    type: 'answer.add_text', text: 'VQA answer', expected_workspace_revision: 1,
  });
  await act(async () => { jest.advanceTimersByTime(15_000); });

  await expect(add).rejects.toThrow(/timed out/i);
  expect(window.answerWorkspaceProbe.pendingAction).toBeNull();
});

test('reconnects with exponential backoff and rehydrates the latest snapshot', async () => {
  renderProvider();
  await openSocket();
  await waitFor(() => expect(getAnswerWorkspace).toHaveBeenCalledTimes(1));
  jest.useFakeTimers();
  await act(async () => MockWebSocket.instances[0].close());
  await act(async () => { jest.advanceTimersByTime(999); });
  expect(MockWebSocket.instances).toHaveLength(1);

  await act(async () => { jest.advanceTimersByTime(1); });
  expect(MockWebSocket.instances).toHaveLength(2);
  await openSocket(1);
  await waitFor(() => expect(getAnswerWorkspace).toHaveBeenCalledTimes(2));
});
