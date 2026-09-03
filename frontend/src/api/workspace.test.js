import {
  createQueryHistory,
  getQueryHistory,
  getSubmissionFiles,
  markFrameViewed,
  markFramesSubmitted,
  workspaceWebSocketUrl,
} from './workspace';

jest.mock('./client', () => {
  const actual = jest.requireActual('./client');
  return { ...actual, requestJson: jest.fn() };
});

import { requestJson } from './client';

beforeEach(() => {
  requestJson.mockReset();
});

test('creates the exact minimal history body', async () => {
  requestJson.mockResolvedValueOnce({ ok: true });
  const snapshot = { results: [{ frame_id: 'frame-1', score: 0.9, frame_ids: ['frame-1'] }] };

  await createQueryHistory({
    queryId: 'query-1',
    userId: 'team A',
    queryText: 'a query',
    resultSnapshot: snapshot,
  });

  expect(requestJson).toHaveBeenCalledWith('/api/v1/query-history', expect.objectContaining({
    method: 'POST',
    body: {
      query_id: 'query-1',
      user_id: 'team A',
      query_text: 'a query',
      result_snapshot: snapshot,
    },
  }));
});

test('encodes the user id in the history URL', async () => {
  requestJson.mockResolvedValueOnce({ items: [] });
  await getQueryHistory({ userId: 'team A' });
  expect(requestJson).toHaveBeenCalledWith('/api/v1/query-history?user_id=team%20A', { signal: undefined });
});

test('sends canonical viewed and submitted activity bodies', async () => {
  requestJson.mockResolvedValue({ ok: true });
  await markFrameViewed({ queryId: 'q/1', frameId: 'frame-1' });
  await markFramesSubmitted({
    queryId: 'q/1',
    submissionFileName: 'query.csv',
    submissionLine: 'V01,20',
    frameIds: ['frame-2', 'frame-3'],
  });

  expect(requestJson).toHaveBeenNthCalledWith(
    1,
    '/api/v1/query-history/q%2F1/viewed-frame',
    { method: 'PATCH', body: { frame_id: 'frame-1' }, signal: undefined },
  );
  expect(requestJson).toHaveBeenNthCalledWith(
    2,
    '/api/v1/query-history/q%2F1/submission',
    {
      method: 'PATCH',
      body: {
        submission_file_name: 'query.csv',
        submission_line: 'V01,20',
        frame_ids: ['frame-2', 'frame-3'],
      },
      signal: undefined,
    },
  );
});

test('validates the shared file response and returns only its shared fields', async () => {
  requestJson.mockResolvedValueOnce({ files: [{
    name: 'query.csv', content: '', is_validated: false, revision: 0, ignored: true,
  }] });
  await expect(getSubmissionFiles()).resolves.toEqual({
    files: [{ name: 'query.csv', content: '', is_validated: false, revision: 0 }],
  });
});

test('rejects invalid values before making a request', async () => {
  await expect(getQueryHistory({ userId: ' ' })).rejects.toThrow('userId');
  await expect(markFramesSubmitted({
    queryId: 'q', submissionFileName: 'x.csv', submissionLine: 'x,1', frameIds: [''],
  })).rejects.toThrow('frameIds[0]');
  expect(requestJson).not.toHaveBeenCalled();
});

test('derives the workspace socket protocol from the configured API base', () => {
  const original = process.env.REACT_APP_API_BASE_URL;
  try {
    jest.resetModules();
    process.env.REACT_APP_API_BASE_URL = 'http://api.example.test/base';
    const httpWorkspace = require('./workspace');
    expect(httpWorkspace.workspaceWebSocketUrl()).toBe('ws://api.example.test/api/v1/workspace/ws');

    jest.resetModules();
    process.env.REACT_APP_API_BASE_URL = 'https://api.example.test/base';
    const httpsWorkspace = require('./workspace');
    expect(httpsWorkspace.workspaceWebSocketUrl()).toBe('wss://api.example.test/api/v1/workspace/ws');
  } finally {
    if (original === undefined) delete process.env.REACT_APP_API_BASE_URL;
    else process.env.REACT_APP_API_BASE_URL = original;
  }
});
