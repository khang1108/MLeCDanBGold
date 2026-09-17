import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import { getCurrentDresTask, submitDresAnswer } from "./api/submissions";
import { openEventTrail, closeEventTrail } from "./api/eventTrail";

const jsonResponse = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

jest.mock("./api/eventTrail", () => ({
  openEventTrail: jest.fn(),
  getEventTrail: jest.fn(),
  actOnEventTrail: jest.fn(),
  closeEventTrail: jest.fn(),
}));

jest.mock("./features/search/components/SearchWorkspace", () => (
  function FakeUnifiedWorkspace({ onFrameClick, onOpenSubmission, replayRequest, userId, historyUserId, onQueryChange, onEventTrailInvalidated }) {
    const frame = {
      frame_id: "f1",
      video_id: "V01",
      frame_idx: 125,
      timestamp_ms: 5000,
    };
    return (
      <div>
        Unified search workspace
        <button
          type="button"
          onClick={() => onFrameClick({
            frame,
            eventTrailContext: {
              snapshotId: 'snap_1',
              resultId: 'r_1',
              kisRevision: 1,
              events: [{ id: 'E1', text: 'woman enters' }],
              searchSessionId: 'q1',
            },
          })}
        >
          Open inspector
        </button>
        <button
          type="button"
          onClick={() => onFrameClick({
            frame: { ...frame, frame_id: 'f2', result_id: 'r_2' },
            eventTrailContext: {
              snapshotId: 'snap_1',
              resultId: 'r_2',
              kisRevision: 1,
              events: [{ id: 'E1', text: 'woman enters' }],
              searchSessionId: 'q1',
            },
          })}
        >
          Open inspector result 2
        </button>
        <button
          type="button"
          onClick={() => onEventTrailInvalidated?.()}
        >
          Invalidate trail
        </button>
        <button
          type="button"
          onClick={() => onQueryChange?.('committed search clue')}
        >
          Set committed query
        </button>
        <button
          type="button"
          onClick={() => onOpenSubmission?.({ videoId: 'V01', startMs: 5000, endMs: 5000 })}
        >
          Submit result to DRES
        </button>
        <output data-testid="replay-request">
          {replayRequest?.item?.query_id || ''}
        </output>
        <output data-testid="query-user-id">{userId || ''}</output>
        <output data-testid="query-history-user-id">{historyUserId || ''}</output>
      </div>
    );
  }
));

jest.mock("./features/workspace/components/WorkspacePage", () => (
  function FakeWorkspacePage({ onReplay, userId, historyRefreshToken }) {
    return (
      <div data-testid="workspace-page">
        Workspace page for {userId}
        <output data-testid="workspace-refresh-token">{historyRefreshToken}</output>
        <button
          type="button"
          onClick={() => onReplay?.({
            query_id: 'saved-query-1',
            query_text: 'saved query',
            result_snapshot: { results: [] },
            frame_activity: { viewed_frame_ids: [] },
          })}
        >
          Replay saved query
        </button>
      </div>
    );
  }
));
jest.mock("./api/submissions", () => ({
  getCurrentDresTask: jest.fn(),
  submitDresAnswer: jest.fn(),
}));
jest.mock("./features/health/hooks/useHealthCheck", () => ({
  useHealthCheck: () => ({ isHealthy: true, healthData: {} }),
}));


beforeEach(() => {
  localStorage.clear();
  getCurrentDresTask.mockReset().mockResolvedValue({
    user_id: 'team-a',
    evaluation_id: 'eval-1',
    task_scope_key: 'dres-task-v1:key-1',
    task_name: 'KIS task', task_group: 'KIS', task_type: 'KIS', duration: 300,
  });
  submitDresAnswer.mockReset().mockResolvedValue({
    state: 'RECORDED', recorded: true, verdict: 'CORRECT', message: 'Recorded by DRES.',
  });
  jest.spyOn(global, 'fetch').mockImplementation(async (_url, options = {}) => {
    if (String(_url).includes('/api/v1/vbs/session/')) {
      return jsonResponse({ user_id: 'team-a', connected: options.method !== 'DELETE' });
    }
    return jsonResponse({ files: [] });
  });
});

afterEach(() => {
  jest.restoreAllMocks();
});

test("mounts one shared search workspace without task tabs", () => {
  render(<App />);

  expect(screen.getByText("Unified search workspace")).toBeTruthy();
  expect(screen.queryByRole("navigation", { name: "Task selection" })).toBeNull();
  expect(screen.getByText(/connect a VBS participant before submitting/i)).toBeTruthy();
});

test("the frame inspector has no direct legacy submission action", () => {
  render(<App />);

  fireEvent.click(screen.getByRole("button", { name: "Open inspector" }));
  expect(screen.queryByRole("button", { name: /submit current frame/i })).toBeNull();
});

test("inspector receives committed query context when frame inspector is opened", () => {
  render(<App />);

  fireEvent.click(screen.getByRole("button", { name: "Set committed query" }));
  fireEvent.click(screen.getByRole("button", { name: "Open inspector" }));
  expect(screen.getByText("committed search clue")).toBeTruthy();
});

test('opens the editable direct-submission popup after participant connection and sends once', async () => {
  render(<App />);
  const userId = screen.getByLabelText('User ID');
  fireEvent.change(userId, { target: { value: 'team-a' } });
  fireEvent.keyDown(userId, { key: 'Enter', code: 'Enter' });
  await screen.findByRole('button', { name: 'OK' });
  await waitFor(() => expect(screen.getByTestId('query-user-id').textContent).toBe('team-a'));

  fireEvent.click(screen.getByRole('button', { name: 'Submit result to DRES' }));

  const dialog = await screen.findByRole('dialog', { name: 'Submit one answer' });
  const answer = screen.getByRole('textbox', { name: 'Answer' });
  expect(answer.value).toBe('V01,5000,5000');
  expect(screen.queryByText(/Submit all/i)).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Submit' }));

  await waitFor(() => expect(submitDresAnswer).toHaveBeenCalledTimes(1));
  expect(submitDresAnswer).toHaveBeenCalledWith({
    userId: 'team-a',
    expectedTaskScopeKey: 'dres-task-v1:key-1',
    evaluationId: 'eval-1',
    taskName: 'KIS task',
    answer: { kind: 'TEMPORAL', video_id: 'V01', start_ms: 5000, end_ms: 5000 },
  });
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  expect(dialog).toBeTruthy();
});

test('clears the connected participant after DRES rejects the private session', async () => {
  submitDresAnswer.mockResolvedValueOnce({
    state: 'NOT_RECORDED',
    recorded: false,
    verdict: null,
    reason: 'DRES_AUTH_REJECTED',
    message: 'DRES rejected this participant session; reconnect before submitting again',
  });
  render(<App />);
  const userId = screen.getByLabelText('User ID');
  fireEvent.change(userId, { target: { value: 'team-a' } });
  fireEvent.keyDown(userId, { key: 'Enter', code: 'Enter' });
  await screen.findByRole('button', { name: 'OK' });
  fireEvent.click(screen.getByRole('button', { name: 'Submit result to DRES' }));
  await screen.findByRole('dialog', { name: 'Submit one answer' });
  fireEvent.click(screen.getByRole('button', { name: 'Submit' }));

  await screen.findByText(/rejected this participant session/i);
  await waitFor(() => expect(userId.disabled).toBe(false));
  expect(localStorage.getItem('hcmai_user_id')).toBeNull();
  expect(screen.getByText(/connect a VBS participant before submitting/i)).toBeTruthy();
});

test('persists and locks the User ID only after the backend handshake', async () => {
  render(<App />);
  const userId = screen.getByLabelText('User ID');
  fireEvent.change(userId, { target: { value: 'team-a' } });

  expect(localStorage.getItem('hcmai_user_id')).toBeNull();
  expect(screen.getByTestId('query-user-id').textContent).toBe('');
  expect(screen.getByTestId('query-history-user-id').textContent).toBe('team-a');
  fireEvent.keyDown(userId, { key: 'Enter', code: 'Enter' });

  await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/vbs/session/connect'),
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ user_id: 'team-a' }),
    }),
  ));
  await waitFor(() => expect(userId.disabled).toBe(true));
  expect(screen.getByRole('button', { name: 'OK' })).toBeTruthy();
  expect(localStorage.getItem('hcmai_user_id')).toBe('team-a');
  expect(screen.getByTestId('query-user-id').textContent).toBe('team-a');

  fireEvent.click(screen.getByRole('button', { name: 'Workspace' }));
  fireEvent.click(screen.getByRole('button', { name: 'Query' }));
  fireEvent.click(screen.getByRole('button', { name: 'Workspace' }));

  expect(userId.value).toBe('team-a');
  expect(screen.getByTestId('workspace-page').textContent).toContain('team-a');
  expect(screen.getByTestId('workspace-refresh-token').textContent).toBe('0');

  fireEvent.click(screen.getByRole('button', { name: 'OK' }));
  await waitFor(() => expect(userId.disabled).toBe(false));
  expect(localStorage.getItem('hcmai_user_id')).toBeNull();
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/vbs/session/team-a'),
    expect.objectContaining({ method: 'DELETE' }),
  );
  expect(screen.getByTestId('query-user-id').textContent).toBe('');
  expect(screen.getByTestId('query-history-user-id').textContent).toBe('team-a');
});

test('replays a saved history item in Query without generating a new request', () => {
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Workspace' }));
  fireEvent.click(screen.getByRole('button', { name: 'Replay saved query' }));

  expect(screen.getByTestId('replay-request').textContent).toContain('saved-query-1');
  expect(screen.getByRole('button', { name: 'Query' }).getAttribute('aria-pressed')).toBe('true');
});

test('navigates to Workspace page when Workspace tab is clicked', () => {
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Workspace' }));
  expect(screen.getByTestId('workspace-page')).toBeTruthy();
});

test('does not display retired tabs (Image Search, Database)', () => {
  render(<App />);
  expect(screen.queryByRole('button', { name: 'Image Search' })).toBeNull();
  expect(screen.queryByRole('button', { name: 'Database' })).toBeNull();
});

test('revalidates a stored User ID before marking the session connected', async () => {
  localStorage.setItem('hcmai_user_id', 'team-stored');
  let resolveStatus;
  global.fetch.mockImplementation((url, options = {}) => {
    if (String(url).includes('/api/v1/vbs/session/team-stored') && options.method === 'GET') {
      return new Promise((resolve) => { resolveStatus = resolve; });
    }
    return jsonResponse({ files: [] });
  });
  render(<App />);

  const userId = screen.getByLabelText('User ID');
  expect(userId.value).toBe('team-stored');
  expect(screen.getAllByText('Connecting…').length).toBeGreaterThan(0);
  expect(screen.getByTestId('query-user-id').textContent).toBe('');
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/vbs/session/team-stored'),
    expect.objectContaining({ method: 'GET' }),
  );

  resolveStatus(jsonResponse({ user_id: 'team-stored', connected: true }));
  await waitFor(() => expect(userId.disabled).toBe(true));
  await waitFor(() => expect(screen.getByTestId('query-user-id').textContent).toBe('team-stored'));
  expect(localStorage.getItem('hcmai_user_id')).toBe('team-stored');
});

test('failed reload reconnection retains history identity but leaves the input unlocked', async () => {
  localStorage.setItem('hcmai_user_id', 'team-stale');
  global.fetch.mockImplementation((url, options = {}) => (
    String(url).includes('/api/v1/vbs/session/team-stale') && options.method === 'GET'
      ? jsonResponse({ detail: 'DRES session is unavailable' }, 503)
      : jsonResponse({ files: [] })
  ));
  render(<App />);

  const userId = screen.getByLabelText('User ID');
  expect(userId.disabled).toBe(true);
  await waitFor(() => expect(userId.disabled).toBe(false));
  expect(userId.value).toBe('team-stale');
  expect(localStorage.getItem('hcmai_user_id')).toBe('team-stale');
  expect(screen.getByTestId('query-user-id').textContent).toBe('');
  expect(screen.getByTestId('query-history-user-id').textContent).toBe('team-stale');
});

test('leaves a failed session editable without persisting or using that identity', async () => {
  global.fetch.mockImplementation((url, options = {}) => (
    String(url).includes('/api/v1/vbs/session/connect')
      ? jsonResponse({ detail: 'VBS user ID has no backend DRES credential mapping' }, 403)
      : jsonResponse({ files: [] })
  ));
  render(<App />);

  const userId = screen.getByLabelText('User ID');
  fireEvent.change(userId, { target: { value: 'unknown-member' } });
  fireEvent.keyDown(userId, { key: 'Enter', code: 'Enter' });

  expect(await screen.findByRole('alert')).toBeTruthy();
  expect(userId.disabled).toBe(false);
  expect(localStorage.getItem('hcmai_user_id')).toBeNull();
  expect(screen.getByTestId('query-user-id').textContent).toBe('');
  expect(screen.getByTestId('query-history-user-id').textContent).toBe('unknown-member');
  expect(screen.queryByLabelText(/password|username|credential/i)).toBeNull();
});

test('manages EventTrail session lifecycle across inspector, modal close, result change, and replay', async () => {
  openEventTrail.mockReset().mockResolvedValue({
    session_id: 'trail_1',
    result_id: 'r_1',
    video_id: 'V01',
    kis_revision: 1,
    trail_revision: 0,
    status: 'active',
    path: [{ event_id: 'E1', frame_id: 'f1', frame_idx: 125, timestamp_ms: 5000 }],
    last_valid_path: null,
    approved_event_ids: [],
    rejected_counts: {},
    window: null,
    submission_selection: null,
    transition: null,
  });
  closeEventTrail.mockReset().mockResolvedValue();

  render(<App />);

  // 1. Open inspector for result 1
  fireEvent.click(screen.getByRole('button', { name: 'Open inspector' }));
  const openTrailBtn = await screen.findByRole('button', { name: /open eventtrail/i });
  fireEvent.click(openTrailBtn);

  await waitFor(() => expect(openEventTrail).toHaveBeenCalledTimes(1));
  expect(openEventTrail).toHaveBeenCalledWith({
    snapshotId: 'snap_1',
    resultId: 'r_1',
    expectedKisRevision: 1,
    searchSessionId: 'q1',
  }, expect.any(Object));

  expect(await screen.findByRole('region', { name: /eventtrail exploration/i })).toBeTruthy();

  // 2. Close modal with Escape -> session preserved, close not called
  fireEvent.keyDown(window, { key: 'Escape' });
  await waitFor(() => expect(screen.queryByRole('region', { name: /eventtrail exploration/i })).toBeNull());
  expect(closeEventTrail).not.toHaveBeenCalled();

  // 3. Reopen same result -> active session reused, openEventTrail NOT called again
  fireEvent.click(screen.getByRole('button', { name: 'Open inspector' }));
  expect(await screen.findByRole('region', { name: /eventtrail exploration/i })).toBeTruthy();
  expect(openEventTrail).toHaveBeenCalledTimes(1);

  // 4. Select different result -> old session closed before opening new result
  fireEvent.click(screen.getByRole('button', { name: 'Open inspector result 2' }));
  await waitFor(() => {
    expect(closeEventTrail).toHaveBeenCalledWith('trail_1', { expectedTrailRevision: 0 });
  });

  // 5. Open trail on result 2
  openEventTrail.mockResolvedValueOnce({
    session_id: 'trail_2',
    result_id: 'r_2',
    video_id: 'V01',
    kis_revision: 1,
    trail_revision: 0,
    status: 'active',
    path: [{ event_id: 'E1', frame_id: 'f2', frame_idx: 130, timestamp_ms: 5200 }],
    last_valid_path: null,
    approved_event_ids: [],
    rejected_counts: {},
    window: null,
    submission_selection: null,
    transition: null,
  });
  const openTrailBtn2 = await screen.findByRole('button', { name: /open eventtrail/i });
  fireEvent.click(openTrailBtn2);
  await screen.findByRole('region', { name: /eventtrail exploration/i });

  // Click 'Exit' -> explicitly closes trail
  const exitBtn = screen.getByRole('button', { name: /^exit$/i });
  fireEvent.click(exitBtn);
  await waitFor(() => {
    expect(closeEventTrail).toHaveBeenCalledWith('trail_2', { expectedTrailRevision: 0 });
    expect(screen.queryByRole('region', { name: /eventtrail exploration/i })).toBeNull();
  });
});

test('keyboard shortcut Windows 1 / Meta+1 and Windows 2 / Meta+2 switch pages', () => {
  render(<App />);

  const queryBtn = screen.getByRole('button', { name: 'Query' });
  const workspaceBtn = screen.getByRole('button', { name: 'Workspace' });

  expect(queryBtn.getAttribute('aria-pressed')).toBe('true');
  expect(workspaceBtn.getAttribute('aria-pressed')).toBe('false');

  // Press Windows 2 / Meta+2 -> switch to Workspace
  fireEvent.keyDown(window, { key: '2', code: 'Digit2', metaKey: true });
  expect(queryBtn.getAttribute('aria-pressed')).toBe('false');
  expect(workspaceBtn.getAttribute('aria-pressed')).toBe('true');

  // Press Windows 1 / Meta+1 -> switch back to Query
  fireEvent.keyDown(window, { key: '1', code: 'Digit1', metaKey: true });
  expect(queryBtn.getAttribute('aria-pressed')).toBe('true');
  expect(workspaceBtn.getAttribute('aria-pressed')).toBe('false');
});

test('keyboard shortcut Ctrl+I focuses User ID input', () => {
  render(<App />);
  const userIdInput = screen.getByLabelText('User ID');
  expect(document.activeElement).not.toBe(userIdInput);

  // Press Ctrl+I
  fireEvent.keyDown(window, { key: 'i', code: 'KeyI', ctrlKey: true });
  expect(document.activeElement).toBe(userIdInput);
});


