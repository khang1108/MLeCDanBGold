import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { getQueryHistory, getSubmissionFiles } from '../../../api/workspace';
import WorkspacePage from './WorkspacePage';
import { SubmissionProvider } from '../../submission/contexts/SubmissionContext';
import { SubmissionDialogProvider } from '../../submission/contexts/SubmissionDialogContext';

jest.mock('../../../api/workspace', () => ({
  getQueryHistory: jest.fn(),
  getSubmissionFiles: jest.fn().mockResolvedValue({ files: [] }),
  workspaceWebSocketUrl: jest.fn(() => 'ws://example.test/api/v1/workspace/ws'),
}));

const historyItem = {
  query_id: 'q1',
  query_text: 'a red vehicle passes',
  submission_files: ['query.csv'],
  result_snapshot: { results: [{ frame_id: 'f1', score: 0.9, frame_ids: ['f1'] }] },
  frame_activity: { viewed_frame_ids: [], submitted_frame_ids: [] },
};

const renderPage = async (props = {}) => {
  const result = render(
    <SubmissionProvider>
      <SubmissionDialogProvider>
        <WorkspacePage {...props} />
      </SubmissionDialogProvider>
    </SubmissionProvider>,
  );
  await act(async () => Promise.resolve());
  return result;
};

beforeEach(() => {
  jest.clearAllMocks();
  getQueryHistory.mockResolvedValue({ items: [historyItem] });
  getSubmissionFiles.mockResolvedValue({ files: [] });
});

test('does not load history without a user id', async () => {
  await renderPage({ isActive: true, userId: '' });
  expect(screen.getByText(/enter a user id in the header/i)).toBeTruthy();
  expect(getQueryHistory).not.toHaveBeenCalled();
});

test('loads bounded history rows and exposes only replay-safe summary fields', async () => {
  await renderPage({ isActive: true, userId: 'team A' });
  expect(await screen.findByText('a red vehicle passes')).toBeTruthy();
  expect(screen.getByText('query.csv')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Replay in Query' })).toBeTruthy();
  expect(screen.queryByText('Query archive')).toBeNull();
  expect(screen.queryByText('Direct inspection')).toBeNull();
  expect(screen.queryByText(/latency|rank|caption|activity|actor/i)).toBeNull();
  expect(getQueryHistory).toHaveBeenCalledWith(expect.objectContaining({ userId: 'team A' }));
});

test('replay forwards the unchanged history item', async () => {
  const onReplay = jest.fn();
  await renderPage({ isActive: true, userId: 'team-a', onReplay });
  fireEvent.click(await screen.findByRole('button', { name: 'Replay in Query' }));
  expect(onReplay).toHaveBeenCalledWith(historyItem);
});

test('validates manual video input without a retrieval or history request', async () => {
  const onOpenManualVideo = jest.fn();
  await renderPage({ isActive: true, userId: '', onOpenManualVideo });
  fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));
  expect(screen.getAllByRole('alert')[0].textContent).toMatch(/video_id/i);
  fireEvent.change(screen.getByLabelText('video_id'), { target: { value: 'L21_V001' } });
  fireEvent.change(screen.getByLabelText('timestamp_ms'), { target: { value: '12000' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));
  await waitFor(() => expect(onOpenManualVideo).toHaveBeenCalledWith({ video_id: 'L21_V001', timestamp_ms: 12000 }));
  expect(getQueryHistory).not.toHaveBeenCalled();
});

test('mounts the shared file worktree in the right column', async () => {
  await renderPage({ isActive: true });
  expect(screen.getByRole('region', { name: 'Shared submission files' })).toBeTruthy();
});
