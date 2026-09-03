import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { getSubmissionFiles } from '../../../api/workspace';
import { SubmissionProvider } from '../contexts/SubmissionContext';
import { SubmissionDialogProvider } from '../contexts/SubmissionDialogContext';
import SubmissionWorktree from './SubmissionWorktree';
import * as submissionArchive from '../submissionArchive';

jest.mock('../../../api/workspace', () => ({
  getSubmissionFiles: jest.fn(),
  workspaceWebSocketUrl: jest.fn(() => 'ws://example.test/api/v1/workspace/ws'),
  markFramesSubmitted: jest.fn(),
}));

class MockWebSocket {
  static instances = [];
  static OPEN = 1;

  constructor() {
    this.readyState = 0;
    this.sent = [];
    MockWebSocket.instances.push(this);
  }

  send(value) { this.sent.push(value); }
  open() { this.readyState = 1; this.onopen?.(); }
  message(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }
  close() { this.readyState = 3; this.onclose?.(); }
}

const renderWorktree = async (files = []) => {
  // SubmissionProvider hydrates on mount and once more after the socket opens.
  // Keep the fixture stable across both calls so the second hydration does not
  // replace the test data with the default empty response.
  getSubmissionFiles.mockResolvedValue({ files });
  const result = render(
    <SubmissionProvider>
      <SubmissionDialogProvider><SubmissionWorktree /></SubmissionDialogProvider>
    </SubmissionProvider>,
  );
  await act(async () => Promise.resolve());
  await act(async () => MockWebSocket.instances[MockWebSocket.instances.length - 1].open());
  await act(async () => Promise.resolve());
  return result;
};

beforeEach(() => {
  jest.clearAllMocks();
  MockWebSocket.instances = [];
  window.WebSocket = MockWebSocket;
  getSubmissionFiles.mockResolvedValue({ files: [] });
});

afterEach(() => {
  delete window.WebSocket;
});

const commitCreate = async (name) => {
  const socket = MockWebSocket.instances[MockWebSocket.instances.length - 1];
  await waitFor(() => expect(socket.sent.some((value) => JSON.parse(value).name === name)).toBe(true));
  await act(async () => socket.message({
    type: 'submission_file.created',
    file: { name, content: '', is_validated: false, revision: 0 },
  }));
};

test('renders the filename-only empty state without browser storage', async () => {
  await renderWorktree();
  expect(screen.getByText('Submission Files')).toBeTruthy();
  expect(screen.getByText('No Query Files')).toBeTruthy();
});

test('creates unique CSV targets from uploaded query files after broadcasts', async () => {
  await renderWorktree();
  fireEvent.change(screen.getByTestId('query-file-input'), {
    target: { files: [
      new File(['one'], 'query_1.txt', { type: 'text/plain' }),
      new File(['two'], 'query_2.txt', { type: 'text/plain' }),
    ] },
  });
  await commitCreate('query_1.csv');
  await commitCreate('query_2.csv');
  expect(await screen.findByText('query_1.csv')).toBeTruthy();
  expect(screen.getByText('query_2.csv')).toBeTruthy();
  expect(screen.queryByText(/lines|empty|has entries/i)).toBeNull();
  expect(screen.getByRole('button', { name: /download csv zip \(0\)/i }).disabled).toBe(true);
});

test('keeps file picker and folder picker fallbacks', async () => {
  const originalPicker = window.showOpenFilePicker;
  window.showOpenFilePicker = jest.fn().mockResolvedValue([
    { getFile: jest.fn().mockResolvedValue(new File(['query'], 'query.txt')) },
  ]);
  await renderWorktree();
  fireEvent.click(screen.getByRole('button', { name: /upload query files/i }));
  await commitCreate('query.csv');
  expect(await screen.findByText('query.csv')).toBeTruthy();
  expect(window.showOpenFilePicker).toHaveBeenCalledTimes(1);
  if (originalPicker) window.showOpenFilePicker = originalPicker;
  else delete window.showOpenFilePicker;
});

test('skips duplicate targets instead of overwriting them', async () => {
  await renderWorktree([{ name: 'query.csv', content: '', is_validated: false, revision: 0 }]);
  await screen.findByText('query.csv');
  fireEvent.change(screen.getByTestId('query-file-input'), {
    target: { files: [new File(['query'], 'query.txt')] },
  });
  expect((await screen.findByRole('status')).textContent).toMatch(/skipped existing/i);
  expect(MockWebSocket.instances[0].sent).toHaveLength(0);
});

test('renders border-only visual state and opens the global editor on one click', async () => {
  await renderWorktree([
    { name: 'empty.csv', content: '', is_validated: false, revision: 0 },
    { name: 'filled.csv', content: 'V01,1', is_validated: false, revision: 1 },
    { name: 'validated.csv', content: 'V01,2', is_validated: true, revision: 2 },
  ]);
  expect(await screen.findByText('empty.csv')).toBeTruthy();
  expect(screen.getByText('empty.csv').parentElement.className).toContain('empty');
  expect(screen.getByText('filled.csv').parentElement.className).toContain('filled');
  expect(screen.getByText('validated.csv').parentElement.className).toContain('validated');
  fireEvent.click(screen.getByRole('button', { name: 'filled.csv' }));
  expect(await screen.findByRole('textbox', { name: /edit filled\.csv content/i })).toBeTruthy();
});

test('downloads only non-empty shared files', async () => {
  jest.spyOn(submissionArchive, 'downloadCsvArchive').mockReturnValue(true);
  await renderWorktree([
    { name: 'filled.csv', content: 'V01,100', is_validated: false, revision: 1 },
    { name: 'empty.csv', content: '  \n', is_validated: false, revision: 2 },
  ]);
  fireEvent.click(await screen.findByRole('button', { name: /download csv zip \(1\)/i }));
  expect(submissionArchive.downloadCsvArchive).toHaveBeenCalledWith([
    { name: 'filled.csv', content: 'V01,100', is_validated: false, revision: 1 },
  ]);
});
