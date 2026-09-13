jest.mock('./client', () => ({ requestJson: jest.fn() }));

import { requestJson } from './client';
import {
  actOnExploration,
  closeExploration,
  getExploration,
  openExploration,
} from './exploration';

afterEach(() => jest.clearAllMocks());

test('uses the exploration transport contract and forwards cancellation signals', async () => {
  const signal = new AbortController().signal;
  requestJson.mockResolvedValue({ handle: 'branch-1' });

  await openExploration({ query: 'boat' }, { signal });
  await getExploration('branch 1', { signal });
  await actOnExploration('branch 1', { action: 'undo' }, { signal });
  await closeExploration('branch 1', 7, { signal });

  expect(requestJson).toHaveBeenNthCalledWith(1, '/api/v1/exploration', {
    method: 'POST', body: { query: 'boat' }, signal,
  });
  expect(requestJson).toHaveBeenNthCalledWith(2, '/api/v1/exploration/branch%201', { signal });
  expect(requestJson).toHaveBeenNthCalledWith(3, '/api/v1/exploration/branch%201/actions', {
    method: 'POST', body: { action: 'undo' }, signal,
  });
  expect(requestJson).toHaveBeenNthCalledWith(4, '/api/v1/exploration/branch%201?expected_revision=7', {
    method: 'DELETE', signal,
  });
});
