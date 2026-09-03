import React from 'react';
import { act, render, screen } from '@testing-library/react';
import { getSubmissionFiles } from '../../../api/workspace';
import { SubmissionProvider, useSubmission } from './SubmissionContext';

jest.mock('../../../api/workspace', () => ({
  getSubmissionFiles: jest.fn(),
  workspaceWebSocketUrl: jest.fn(() => 'ws://example.test/api/v1/workspace/ws'),
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

  open() {
    this.readyState = 1;
    this.onopen?.();
  }

  message(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }

  close() {
    this.readyState = 3;
    this.onclose?.();
  }
}

const Probe = () => {
  const context = useSubmission();
  window.submissionProbe = context;
  return <output data-testid="file-names">{context.files.map((file) => file.name).join(',')}</output>;
};

const renderProvider = async ({ flushInitial = true } = {}) => {
  const result = render(<SubmissionProvider><Probe /></SubmissionProvider>);
  if (flushInitial) await act(async () => Promise.resolve());
  return result;
};

const openSocket = async (index = 0) => {
  await act(async () => {
    MockWebSocket.instances[index].open();
    await Promise.resolve();
  });
};

beforeEach(() => {
  jest.useFakeTimers();
  jest.clearAllMocks();
  MockWebSocket.instances = [];
  global.WebSocket = MockWebSocket;
  getSubmissionFiles.mockResolvedValue({ files: [] });
});

afterEach(() => {
  delete global.WebSocket;
  delete window.submissionProbe;
  jest.useRealTimers();
});

test('connects globally without touching browser storage and hydrates after open', async () => {
  const getItem = jest.spyOn(Storage.prototype, 'getItem');
  const setItem = jest.spyOn(Storage.prototype, 'setItem');
  getSubmissionFiles.mockResolvedValue({ files: [{ name: 'a.csv', content: '', is_validated: false, revision: 0 }] });
  await renderProvider();
  expect(MockWebSocket.instances).toHaveLength(1);
  expect(getItem).not.toHaveBeenCalled();
  expect(setItem).not.toHaveBeenCalled();
  await openSocket();
  expect((await screen.findByTestId('file-names')).textContent).toContain('a.csv');
  getItem.mockRestore();
  setItem.mockRestore();
});

test('sorts worktree filenames by their embedded numeric order', async () => {
  getSubmissionFiles.mockResolvedValue({ files: [
    { name: 'query-team-10-final.csv', content: '', is_validated: false, revision: 0 },
    { name: 'query-team-2-final.csv', content: '', is_validated: false, revision: 0 },
    { name: 'query-team-1-final.csv', content: '', is_validated: false, revision: 0 },
  ] });
  await renderProvider();
  await openSocket();

  expect(screen.getByTestId('file-names').textContent).toBe(
    'query-team-1-final.csv,query-team-2-final.csv,query-team-10-final.csv',
  );
});

test('replays file events received while hydration is pending', async () => {
  let resolveHydration;
  getSubmissionFiles.mockReturnValueOnce(new Promise((resolve) => { resolveHydration = resolve; }));
  await renderProvider({ flushInitial: false });
  await openSocket();
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.updated',
    file: { name: 'a.csv', content: 'new', is_validated: false, revision: 2 },
  }));
  await act(async () => {
    resolveHydration({ files: [{ name: 'a.csv', content: 'old', is_validated: false, revision: 1 }] });
    await Promise.resolve();
  });
  expect(screen.getByTestId('file-names').textContent).toContain('a.csv');
  expect(window.submissionProbe.files[0].content).toBe('new');
});

test('does not regress a newer file and delete wins a late hydration response', async () => {
  let resolveHydration;
  getSubmissionFiles.mockReturnValueOnce(new Promise((resolve) => { resolveHydration = resolve; }));
  await renderProvider({ flushInitial: false });
  await openSocket();
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.updated',
    file: { name: 'a.csv', content: 'new', is_validated: false, revision: 3 },
  }));
  await act(async () => MockWebSocket.instances[0].message({ type: 'submission_file.deleted', name: 'a.csv' }));
  await act(async () => {
    resolveHydration({ files: [{ name: 'a.csv', content: 'old', is_validated: false, revision: 1 }] });
    await Promise.resolve();
  });
  expect(screen.getByTestId('file-names').textContent).toBe('');
});

test('reconnects with a fresh hydration after the bounded retry delay', async () => {
  getSubmissionFiles
    // The first response belongs to the mount-time hydration; the next two
    // belong to the initial and reconnect socket opens respectively.
    .mockResolvedValueOnce({ files: [] })
    .mockResolvedValueOnce({ files: [{ name: 'a.csv', content: '', is_validated: false, revision: 0 }] })
    .mockResolvedValueOnce({ files: [{ name: 'b.csv', content: '', is_validated: false, revision: 0 }] });
  await renderProvider();
  await openSocket();
  await act(async () => MockWebSocket.instances[0].close());
  expect(MockWebSocket.instances).toHaveLength(1);
  await act(async () => jest.advanceTimersByTime(999));
  expect(MockWebSocket.instances).toHaveLength(1);
  await act(async () => jest.advanceTimersByTime(1));
  expect(MockWebSocket.instances).toHaveLength(2);
  await openSocket(1);
  expect(screen.getByTestId('file-names').textContent).toBe('b.csv');
  expect(getSubmissionFiles).toHaveBeenCalledTimes(3);
});

test('sends a revision-aware update and resolves only on the matching broadcast', async () => {
  getSubmissionFiles.mockResolvedValueOnce({ files: [{ name: 'a.csv', content: 'old', is_validated: false, revision: 3 }] });
  await renderProvider();
  await openSocket();
  let result;
  await act(async () => {
    result = window.submissionProbe.updateFile({ name: 'a.csv', content: 'new', expectedRevision: 3 });
  });
  expect(JSON.parse(MockWebSocket.instances[0].sent[0])).toEqual({
    type: 'submission_file.update', name: 'a.csv', content: 'new', expected_revision: 3,
  });
  let resolved = false;
  result.then(() => { resolved = true; });
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.updated',
    file: { name: 'a.csv', content: 'new', is_validated: false, revision: 4 },
  }));
  await expect(result).resolves.toMatchObject({ revision: 4 });
  expect(resolved).toBe(true);
});

test('conflict updates the mirror and rejects without retrying', async () => {
  getSubmissionFiles.mockResolvedValueOnce({ files: [{ name: 'a.csv', content: 'old', is_validated: false, revision: 1 }] });
  await renderProvider();
  await openSocket();
  let pending;
  await act(async () => {
    pending = window.submissionProbe.updateFile({ name: 'a.csv', content: 'mine', expectedRevision: 1 });
  });
  pending.catch(() => {});
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.conflict',
    file: { name: 'a.csv', content: 'theirs', is_validated: false, revision: 2 },
  }));
  await expect(pending).rejects.toMatchObject({ code: 'REVISION_CONFLICT', latestFile: { content: 'theirs' } });
  expect(MockWebSocket.instances[0].sent).toHaveLength(1);
  expect(window.submissionProbe.files[0].content).toBe('theirs');
});

test('create, validate, and delete resolve only from their matching broadcasts', async () => {
  getSubmissionFiles.mockResolvedValueOnce({ files: [{ name: 'a.csv', content: '', is_validated: false, revision: 1 }] });
  await renderProvider();
  await openSocket();

  let create;
  await act(async () => {
    create = window.submissionProbe.createFile({ name: 'b.csv', content: '' });
  });
  expect(JSON.parse(MockWebSocket.instances[0].sent.at(-1))).toEqual({
    type: 'submission_file.create', name: 'b.csv', content: '',
  });
  let created = false;
  create.then(() => { created = true; });
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.updated',
    file: { name: 'b.csv', content: '', is_validated: false, revision: 1 },
  }));
  expect(created).toBe(false);
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.created',
    file: { name: 'b.csv', content: '', is_validated: false, revision: 0 },
  }));
  await expect(create).resolves.toMatchObject({ name: 'b.csv' });

  let validate;
  await act(async () => {
    validate = window.submissionProbe.validateFile({ name: 'a.csv', expectedRevision: 1 });
  });
  expect(JSON.parse(MockWebSocket.instances[0].sent.at(-1))).toEqual({
    type: 'submission_file.validate', name: 'a.csv', is_validated: true, expected_revision: 1,
  });
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.updated',
    file: { name: 'a.csv', content: '', is_validated: true, revision: 2 },
  }));
  await expect(validate).resolves.toMatchObject({ revision: 2, is_validated: true });

  let remove;
  await act(async () => {
    remove = window.submissionProbe.deleteFile({ name: 'a.csv', expectedRevision: 2 });
  });
  expect(JSON.parse(MockWebSocket.instances[0].sent.at(-1))).toEqual({
    type: 'submission_file.delete', name: 'a.csv', expected_revision: 2,
  });
  await act(async () => MockWebSocket.instances[0].message({
    type: 'submission_file.deleted', name: 'a.csv',
  }));
  await expect(remove).resolves.toMatchObject({ name: 'a.csv' });
});

test('rejects mutations while the workspace socket is not open', async () => {
  await renderProvider();
  await expect(window.submissionProbe.updateFile({
    name: 'a.csv', content: 'V01,1', expectedRevision: 0,
  })).rejects.toThrow(/not connected/i);
  expect(MockWebSocket.instances[0].sent).toHaveLength(0);
});
