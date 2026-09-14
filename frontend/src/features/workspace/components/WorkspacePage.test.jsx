import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { getQueryHistory } from '../../../api/history';
import { resolveFrameAtTimestamp } from '../../../api/frames';
import WorkspacePage from './WorkspacePage';

jest.mock('../../../api/history', () => ({
  getQueryHistory: jest.fn(),
}));
jest.mock('../../../api/frames', () => ({
  resolveFrameAtTimestamp: jest.fn(),
}));

const historyItem = {
  query_id: 'q1',
  query_text: 'a red vehicle passes',
  result_snapshot: { results: [{ frame_id: 'f1', score: 0.9, frame_ids: ['f1'] }] },
  frame_activity: { viewed_frame_ids: [] },
};

const renderPage = async (props = {}) => {
  const result = render(<WorkspacePage {...props} />);
  await act(async () => Promise.resolve());
  return result;
};

beforeEach(() => {
  jest.clearAllMocks();
  getQueryHistory.mockResolvedValue({ items: [historyItem] });
  resolveFrameAtTimestamp.mockResolvedValue({
    requested_timestamp_ms: 12000,
    frame_id: 'L21_V001_00000300',
    video_id: 'L21_V001',
    frame_idx: 300,
    timestamp_ms: 12040,
    fps: 25,
    metadata: { caption: 'A traffic scene', objects: ['traffic'] },
  });
});

test('does not load history without a user id', async () => {
  await renderPage({ isActive: true, userId: '' });
  expect(screen.getByText(/enter a user id in the header/i)).toBeTruthy();
  expect(getQueryHistory).not.toHaveBeenCalled();
});

test('loads bounded history rows and exposes only replay-safe summary fields', async () => {
  await renderPage({ isActive: true, userId: 'team A' });
  expect(await screen.findByText('a red vehicle passes')).toBeTruthy();
  expect(screen.queryByText('query.csv')).toBeNull();
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

test('resolves manual video input before opening the metadata viewer', async () => {
  const onOpenManualVideo = jest.fn();
  await renderPage({ isActive: true, userId: '', onOpenManualVideo });
  fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));
  expect(screen.getAllByRole('alert')[0].textContent).toMatch(/video_id/i);
  fireEvent.change(screen.getByLabelText('video_id'), { target: { value: 'L21_V001' } });
  fireEvent.change(screen.getByLabelText('timestamp_ms'), { target: { value: '12000' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));
  await waitFor(() => expect(resolveFrameAtTimestamp).toHaveBeenCalledWith({
    videoId: 'L21_V001',
    timestampMs: 12000,
    signal: expect.any(AbortSignal),
  }));
  await waitFor(() => expect(onOpenManualVideo).toHaveBeenCalledWith({
    frame: expect.objectContaining({
      frame_id: 'L21_V001_00000300',
      timestamp_ms: 12040,
      metadata: { caption: 'A traffic scene', objects: ['traffic'] },
    }),
    requestedTimestampMs: 12000,
  }));
  expect(getQueryHistory).not.toHaveBeenCalled();
});

test('keeps the viewer closed and exposes a resolver error', async () => {
  const onOpenManualVideo = jest.fn();
  resolveFrameAtTimestamp.mockRejectedValueOnce(new Error('Canonical frame not found'));
  await renderPage({ isActive: true, userId: '', onOpenManualVideo });

  fireEvent.change(screen.getByLabelText('video_id'), { target: { value: 'L21_V001' } });
  fireEvent.change(screen.getByLabelText('timestamp_ms'), { target: { value: '12000' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));

  expect(await screen.findByText('Canonical frame not found')).toBeTruthy();
  expect(onOpenManualVideo).not.toHaveBeenCalled();
});

test('renders inline video player without captions and allows closing it', async () => {
  await renderPage({ isActive: true, userId: '' });
  expect(screen.queryByRole('region', { name: /inline player/i })).toBeNull();

  fireEvent.change(screen.getByLabelText('video_id'), { target: { value: 'L21_V001' } });
  fireEvent.change(screen.getByLabelText('timestamp_ms'), { target: { value: '12000' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));

  const inlinePlayer = await screen.findByRole('region', { name: /inline player for L21_V001/i });
  expect(inlinePlayer).toBeTruthy();
  expect(screen.queryByText('A traffic scene')).toBeNull();
  expect(screen.getByRole('button', { name: /close video player/i })).toBeTruthy();

  fireEvent.click(screen.getByRole('button', { name: /close video player/i }));
  expect(screen.queryByRole('region', { name: /inline player/i })).toBeNull();
});

test('does not mount the legacy submission-file worktree in the right column', async () => {
  await renderPage({ isActive: true });
  expect(screen.queryByRole('region', { name: 'Shared submission files' })).toBeNull();
});
