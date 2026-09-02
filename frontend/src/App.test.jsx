import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

jest.mock("./features/search/components/SearchWorkspace", () => (
  function FakeUnifiedWorkspace({ onFrameClick, replayRequest, userId }) {
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
          onClick={() => onFrameClick({ frame, submissionMode: "kis" })}
        >
          Open KIS inspector
        </button>
        <button
          type="button"
          onClick={() => onFrameClick({ frame, submissionMode: "none" })}
        >
          Open TRAKE inspector
        </button>
        <output data-testid="replay-request">
          {replayRequest?.item?.query_id || ''}
        </output>
        <output data-testid="query-user-id">{userId || ''}</output>
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
            submission_files: [],
            result_snapshot: { results: [] },
            frame_activity: { viewed_frame_ids: [], submitted_frame_ids: [] },
          })}
        >
          Replay saved query
        </button>
        <button
          type="button"
          onClick={() => onOpenManualVideo?.({ video_id: 'V01', timestamp_ms: 1000 })}
        >
          Open manual video
        </button>
      </div>
    );
  }
));
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

test("mounts one shared search workspace without task tabs", () => {
  render(<App />);

  expect(screen.getByText("Unified search workspace")).toBeTruthy();
  expect(screen.queryByRole("navigation", { name: "Task selection" })).toBeNull();
});

test("only KIS frame selections expose inspector submission", () => {
  render(<App />);

  fireEvent.click(screen.getByRole("button", { name: "Open TRAKE inspector" }));
  expect(screen.queryByRole("button", { name: /submit current frame/i })).toBeNull();

  fireEvent.click(screen.getByRole("button", { name: "Close popup" }));
  fireEvent.click(screen.getByRole("button", { name: "Open KIS inspector" }));
  expect(screen.getByRole("button", { name: /submit current frame/i })).toBeTruthy();
});

test('keeps the shared User ID while switching Query, Filter, and Workspace', () => {
  render(<App />);
  const userId = screen.getByLabelText('User ID');
  fireEvent.change(userId, { target: { value: 'team-a' } });

  expect(screen.getByTestId('query-user-id').textContent).toBe('team-a');

  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));
  fireEvent.click(screen.getByRole('button', { name: 'Workspace' }));

  expect(userId.value).toBe('team-a');
  expect(screen.getByTestId('workspace-page').textContent).toContain('team-a');
  expect(screen.getByTestId('workspace-refresh-token').textContent).toBe('0');
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

  expect(screen.getByText('V01 · 1000 ms')).toBeTruthy();
  expect(screen.queryByRole('button', { name: /submit current frame/i })).toBeNull();
});
