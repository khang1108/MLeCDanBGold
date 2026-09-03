import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import {
  getSubmissionFiles,
  markFramesSubmitted,
} from '../../../api/workspace';
import { SubmissionProvider, useSubmission } from './SubmissionContext';
import { SubmissionDialogProvider, useSubmissionDialog } from './SubmissionDialogContext';

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

const Probe = () => {
  const submission = useSubmission();
  const dialog = useSubmissionDialog();
  window.dialogProbe = dialog;
  return (
    <div>
      <button type="button" onClick={() => dialog.requestSubmission({
        line: 'V01,20', source: 'KIS frame', history: { queryId: 'q1', frameIds: ['f1'] },
      })}>Request submission</button>
      <button type="button" onClick={() => dialog.openEditor('a.csv')}>Open editor</button>
      <output>{submission.files.map((file) => file.content).join('|')}</output>
    </div>
  );
};

const renderDialogs = () => render(
  <SubmissionProvider>
    <SubmissionDialogProvider><Probe /></SubmissionDialogProvider>
  </SubmissionProvider>,
);

beforeEach(async () => {
  jest.clearAllMocks();
  MockWebSocket.instances = [];
  global.WebSocket = MockWebSocket;
  getSubmissionFiles.mockResolvedValue({ files: [{ name: 'a.csv', content: '', is_validated: false, revision: 1 }] });
  markFramesSubmitted.mockResolvedValue({ ok: true });
});

afterEach(() => {
  delete global.WebSocket;
  delete window.dialogProbe;
});

const openAndHydrate = async () => {
  renderDialogs();
  await act(async () => Promise.resolve());
  await act(async () => MockWebSocket.instances[0].open());
  await act(async () => Promise.resolve());
};

test('opens an editor with committed content and discards Esc without sending', async () => {
  // The provider hydrates both before and after the socket opens.
  getSubmissionFiles.mockResolvedValue({ files: [{ name: 'a.csv', content: 'V01,1', is_validated: false, revision: 1 }] });
  await openAndHydrate();
  fireEvent.click(screen.getByRole('button', { name: 'Open editor' }));
  const editor = await screen.findByRole('textbox', { name: /edit a\.csv content/i });
  expect(editor.value).toBe('V01,1');
  fireEvent.change(editor, { target: { value: 'V01,2' } });
  fireEvent.keyDown(editor, { key: 'Escape', code: 'Escape' });
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  expect(MockWebSocket.instances[0].sent).toHaveLength(0);
});

test('picker appends an intent to the draft and saves only after the update broadcast', async () => {
  await openAndHydrate();
  fireEvent.click(screen.getByRole('button', { name: 'Request submission' }));
  fireEvent.click(await screen.findByRole('button', { name: 'a.csv' }));
  const editor = await screen.findByRole('textbox', { name: /edit a\.csv content/i });
  expect(editor.value).toBe('V01,20');
  expect(MockWebSocket.instances[0].sent).toHaveLength(0);
  fireEvent.click(screen.getByRole('button', { name: 'Lưu' }));
  expect(JSON.parse(MockWebSocket.instances[0].sent[0])).toEqual({
    type: 'submission_file.update', name: 'a.csv', content: 'V01,20', expected_revision: 1,
  });
  expect(markFramesSubmitted).not.toHaveBeenCalled();
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.updated',
    file: { name: 'a.csv', content: 'V01,20', is_validated: false, revision: 2 },
  }));
  await waitFor(() => expect(markFramesSubmitted).toHaveBeenCalledWith({
    queryId: 'q1', submissionFileName: 'a.csv', submissionLine: 'V01,20', frameIds: ['f1'],
  }));
});

test('closes after a file save without waiting for the history patch', async () => {
  let resolveHistoryPatch;
  markFramesSubmitted.mockImplementationOnce(() => new Promise((resolve) => {
    resolveHistoryPatch = resolve;
  }));
  await openAndHydrate();
  fireEvent.click(screen.getByRole('button', { name: 'Request submission' }));
  fireEvent.click(await screen.findByRole('button', { name: 'a.csv' }));
  const editor = await screen.findByRole('textbox', { name: /edit a\.csv content/i });

  fireEvent.keyDown(editor, { key: 'Enter', code: 'Enter' });
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.updated',
    file: { name: 'a.csv', content: 'V01,20', is_validated: false, revision: 2 },
  }));

  await waitFor(() => expect(markFramesSubmitted).toHaveBeenCalledWith({
    queryId: 'q1', submissionFileName: 'a.csv', submissionLine: 'V01,20', frameIds: ['f1'],
  }));
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());

  await act(async () => resolveHistoryPatch({ ok: true }));
});

test('closes after validation finishes', async () => {
  getSubmissionFiles.mockResolvedValue({ files: [
    { name: 'a.csv', content: 'V01,1', is_validated: false, revision: 1 },
  ] });
  await openAndHydrate();
  fireEvent.click(screen.getByRole('button', { name: 'Open editor' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Validate' }));
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.updated',
    file: { name: 'a.csv', content: '', is_validated: true, revision: 2 },
  }));

  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
});

test('allows the editor to close while save confirmation is pending', async () => {
  await openAndHydrate();
  fireEvent.click(screen.getByRole('button', { name: 'Open editor' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Lưu' }));
  expect(MockWebSocket.instances[0].sent).toHaveLength(1);

  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());

  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.updated',
    file: { name: 'a.csv', content: '', is_validated: false, revision: 2 },
  }));
});

test('shows an explicit conflict and allows rebasing the draft', async () => {
  await openAndHydrate();
  fireEvent.click(screen.getByRole('button', { name: 'Open editor' }));
  const editor = await screen.findByRole('textbox', { name: /edit a\.csv content/i });
  fireEvent.change(editor, { target: { value: 'mine' } });
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.conflict',
    file: { name: 'a.csv', content: 'theirs', is_validated: false, revision: 2 },
  }));
  expect(await screen.findByText('This file changed on the server.')).toBeTruthy();
  expect(editor.value).toBe('mine');
  fireEvent.click(screen.getByRole('button', { name: /keep draft and rebase/i }));
  fireEvent.click(screen.getByRole('button', { name: 'Lưu' }));
  expect(JSON.parse(MockWebSocket.instances[0].sent[0])).toMatchObject({ expected_revision: 2 });
});
