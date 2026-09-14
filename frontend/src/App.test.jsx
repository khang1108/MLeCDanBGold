import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import { answerWorkspaceWebSocketUrl, getAnswerWorkspace } from "./api/answerWorkspace";

const jsonResponse = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

jest.mock("./features/search/components/SearchWorkspace", () => (
  function FakeUnifiedWorkspace({ onFrameClick, onAddCandidate, replayRequest, userId, historyUserId }) {
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
          onClick={() => onFrameClick({ frame })}
        >
          Open inspector
        </button>
        <button
          type="button"
          onClick={() => onAddCandidate?.({ kind: 'FRAME', videoId: 'V01', timestampMs: 5000 })}
        >
          Add result to answer workspace
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
  function FakeWorkspacePage({ onReplay, onOpenManualVideo, userId, historyRefreshToken }) {
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
        <button
          type="button"
          onClick={() => onOpenManualVideo?.({
            frame: {
              frame_id: 'V01_00000025',
              video_id: 'V01',
              frame_idx: 25,
              timestamp_ms: 1040,
              fps: 25,
              metadata: { caption: 'Resolved evidence' },
            },
            requestedTimestampMs: 1000,
          })}
        >
          Open manual video
        </button>
      </div>
    );
  }
));
jest.mock("./features/database", () => ({
  DatabasePage: function FakeDatabasePage({ isActive }) {
    return <div data-testid="database-page">Database Page (active: {String(isActive)})</div>;
  },
}));
jest.mock("./api/answerWorkspace", () => ({
  getAnswerWorkspace: jest.fn(),
  answerWorkspaceWebSocketUrl: jest.fn((userId) => (
    `ws://example.test/api/v1/answer-workspace/ws?user_id=${encodeURIComponent(userId)}`
  )),
  normalizeAnswerWorkspace: jest.requireActual("./api/answerWorkspace").normalizeAnswerWorkspace,
}));
jest.mock("./features/health/hooks/useHealthCheck", () => ({
  useHealthCheck: () => ({ isHealthy: true, healthData: {} }),
}));
jest.mock("./features/vim/hooks/useVimMode", () => ({
  useVimMode: () => ({
    mode: "NORMAL",
    enterInsertMode: jest.fn(),
    enterNormalMode: jest.fn(),
    setMode: jest.fn(),
    isTopKOpen: false,
    setIsTopKOpen: jest.fn(),
    isHelpOpen: false,
    setIsHelpOpen: jest.fn(),
  }),
}));

class MockWebSocket {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    MockWebSocket.instances.push(this);
  }
  close() { this.readyState = 3; this.onclose?.(); }
}

beforeEach(() => {
  localStorage.clear();
  MockWebSocket.instances = [];
  window.WebSocket = MockWebSocket;
  answerWorkspaceWebSocketUrl.mockImplementation((userId) => (
    `ws://example.test/api/v1/answer-workspace/ws?user_id=${encodeURIComponent(userId)}`
  ));
  getAnswerWorkspace.mockResolvedValue({
    avs_enabled: false,
    evaluation_id: 'eval-1',
    task_scope_key: 'dres-task-v1:key-1',
    task_name: 'KIS task',
    revision: 0,
    updated_by_user_id: '',
    updated_at_ms: 0,
    candidates: [],
    pending_submission: null,
    active_evaluation_id: 'eval-1',
    active_task_scope_key: 'dres-task-v1:key-1',
    active_task_name: 'KIS task',
    task_scope_mismatch: false,
  });
  jest.spyOn(global, 'fetch').mockImplementation(async (_url, options = {}) => {
    if (String(_url).includes('/api/v1/vbs/session/')) {
      return jsonResponse({ user_id: 'team-a', connected: options.method !== 'DELETE' });
    }
    return jsonResponse({ files: [] });
  });
});

afterEach(() => {
  delete window.WebSocket;
  jest.restoreAllMocks();
});

test("mounts one shared search workspace without task tabs", () => {
  render(<App />);

  expect(screen.getByText("Unified search workspace")).toBeTruthy();
  expect(screen.queryByRole("navigation", { name: "Task selection" })).toBeNull();
});

test("the frame inspector has no direct legacy submission action", () => {
  render(<App />);

  fireEvent.click(screen.getByRole("button", { name: "Open inspector" }));
  expect(screen.queryByRole("button", { name: /submit current frame/i })).toBeNull();
});

test('opens the editable frame candidate dialog only after the participant handshake', async () => {
  render(<App />);
  const userId = screen.getByLabelText('User ID');
  fireEvent.change(userId, { target: { value: 'team-a' } });
  fireEvent.keyDown(userId, { key: 'Enter', code: 'Enter' });
  await screen.findByRole('button', { name: 'OK' });
  await waitFor(() => expect(screen.getByTestId('query-user-id').textContent).toBe('team-a'));

  fireEvent.click(screen.getByRole('button', { name: 'Add result to answer workspace' }));

  expect(screen.getByRole('dialog', { name: 'Add FRAME answer' })).toBeTruthy();
  expect(screen.getByLabelText('Video ID').value).toBe('V01');
  expect(screen.getByLabelText('Timestamp (ms)').value).toBe('5000');
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

  fireEvent.click(screen.getByRole('button', { name: 'Database' }));
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

test('opens manual video inspection without exposing a fake frame submission', () => {
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Workspace' }));
  fireEvent.click(screen.getByRole('button', { name: 'Open manual video' }));

  expect(screen.getByText('V01 · 25')).toBeTruthy();
  expect(screen.getByText('Resolved evidence')).toBeTruthy();
  expect(screen.queryByRole('button', { name: /submit current frame/i })).toBeNull();
});

test('navigates to Database workspace when Database tab is clicked', () => {
  render(<App />);
  const databaseTab = screen.getByRole('button', { name: 'Database' });
  expect(databaseTab.getAttribute('aria-pressed')).toBe('false');

  fireEvent.click(databaseTab);
  expect(databaseTab.getAttribute('aria-pressed')).toBe('true');
  expect(screen.getByTestId('database-page').textContent).toContain('active: true');
});

test('does not display the retired standalone Image Search tab', () => {
  render(<App />);
  expect(screen.queryByRole('button', { name: 'Image Search' })).toBeNull();
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

