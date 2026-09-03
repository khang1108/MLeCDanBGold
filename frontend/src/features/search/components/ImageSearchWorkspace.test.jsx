import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFramesByImage } from '../../../api/search';
import ImageSearchWorkspace from './ImageSearchWorkspace';
import { SubmissionProvider } from '../../submission/contexts/SubmissionContext';
import { SubmissionDialogProvider } from '../../submission/contexts/SubmissionDialogContext';
import {
  createQueryHistory,
  getSubmissionFiles,
  markFrameViewed,
} from '../../../api/workspace';

jest.mock('../../../api/search');
jest.mock('../../../api/workspace', () => ({
  getSubmissionFiles: jest.fn(),
  workspaceWebSocketUrl: jest.fn(() => 'ws://example.test/api/v1/workspace/ws'),
  createQueryHistory: jest.fn(),
  markFrameViewed: jest.fn(),
  markFramesSubmitted: jest.fn(),
}));

class MockWebSocket {
  static instances = [];
  static OPEN = 1;
  constructor() { this.readyState = 0; this.sent = []; MockWebSocket.instances.push(this); }
  send(value) { this.sent.push(value); }
  open() { this.readyState = 1; this.onopen?.(); }
  message(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }
  close() { this.readyState = 3; this.onclose?.(); }
}

const renderImageSearch = (props) => {
  const result = render(
    <SubmissionProvider>
      <SubmissionDialogProvider>
        <ImageSearchWorkspace {...props} />
      </SubmissionDialogProvider>
    </SubmissionProvider>,
  );
  act(() => MockWebSocket.instances[MockWebSocket.instances.length - 1]?.open?.());
  return result;
};

beforeEach(() => {
  searchFramesByImage.mockReset();
  getSubmissionFiles.mockReturnValue(new Promise(() => {}));
  createQueryHistory.mockResolvedValue({});
  markFrameViewed.mockResolvedValue({});
  MockWebSocket.instances = [];
  window.WebSocket = MockWebSocket;
  global.URL.createObjectURL = jest.fn(() => 'blob:mock-image-preview');
  global.URL.revokeObjectURL = jest.fn();
});

afterEach(() => {
  delete window.WebSocket;
  delete global.URL.createObjectURL;
  delete global.URL.revokeObjectURL;
});

test('renders empty image dropzone with disabled Search button', () => {
  renderImageSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });

  expect(screen.getByText(/Choose or drop an image/i)).toBeTruthy();
  const searchBtn = screen.getByRole('button', { name: 'Search' });
  expect(searchBtn.disabled).toBe(true);
});

test('selecting an image displays preview and enables Search button', async () => {
  renderImageSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });

  const fileInput = document.querySelector('input[type="file"]');
  const file = new File(['dummy-image'], 'kitchen.png', { type: 'image/png' });

  fireEvent.change(fileInput, { target: { files: [file] } });

  expect(screen.getByText('kitchen.png')).toBeTruthy();
  const searchBtn = screen.getByRole('button', { name: 'Search' });
  expect(searchBtn.disabled).toBe(false);
});

test('clearing the selected image resets preview and disables Search button', () => {
  renderImageSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });

  const fileInput = document.querySelector('input[type="file"]');
  const file = new File(['dummy-image'], 'kitchen.png', { type: 'image/png' });

  fireEvent.change(fileInput, { target: { files: [file] } });
  expect(screen.getByText('kitchen.png')).toBeTruthy();

  const clearBtn = screen.getByTitle(/Remove image/i);
  fireEvent.click(clearBtn);

  expect(screen.queryByText('kitchen.png')).toBeNull();
  expect(screen.getByRole('button', { name: 'Search' }).disabled).toBe(true);
});

test('submitting search calls searchFramesByImage and renders results with latency', async () => {
  const mockResults = [
    {
      frame_id: 'img-frame-1',
      video_id: 'L01_V001',
      frame_idx: 120,
      timestamp_ms: 4800,
      score: 0.95,
      frame_ids: ['img-frame-1'],
      timestamps_ms: [4800],
      metadata: { caption: 'Chef cooking in restaurant' },
    },
  ];
  searchFramesByImage.mockResolvedValue({
    results: mockResults,
    latency: { query_ms: 12, retrieval_ms: 30, alignment_ms: 0, materialization_ms: 5, total_ms: 47 },
  });

  const onHistoryRefresh = jest.fn();
  const onFrameClick = jest.fn();

  renderImageSearch({
    topK: 15,
    setTopK: jest.fn(),
    userId: 'team-a',
    onHistoryRefresh,
    onFrameClick,
  });

  const fileInput = document.querySelector('input[type="file"]');
  const file = new File(['dummy-image'], 'chef.jpg', { type: 'image/jpeg' });
  fireEvent.change(fileInput, { target: { files: [file] } });

  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  await waitFor(() => {
    expect(searchFramesByImage).toHaveBeenCalledWith({
      imageFile: file,
      topK: 15,
      signal: expect.any(AbortSignal),
    });
  });

  await waitFor(() => {
    expect(document.querySelector('.latency-summary')?.textContent).toMatch(/Found\s+1\s+frames in\s+47ms/);
  });

  expect(createQueryHistory).toHaveBeenCalledWith(expect.objectContaining({
    userId: 'team-a',
    queryText: '[Image] chef.jpg',
  }));
  expect(onHistoryRefresh).toHaveBeenCalled();

  // Click frame
  const frameCard = screen.getByText(/L01_V001/).closest('.frame-card');
  fireEvent.click(frameCard);
  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    frame: mockResults[0],
    submissionMode: 'kis',
  }));
});

test('prompts for User ID when searching without one', async () => {
  const onFocusUserId = jest.fn();
  renderImageSearch({
    topK: 20,
    setTopK: jest.fn(),
    userId: '',
    onFocusUserId,
  });

  const fileInput = document.querySelector('input[type="file"]');
  const file = new File(['dummy-image'], 'test.png', { type: 'image/png' });
  fireEvent.change(fileInput, { target: { files: [file] } });

  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  expect(onFocusUserId).toHaveBeenCalled();
  expect(searchFramesByImage).not.toHaveBeenCalled();
  expect(screen.getByText(/Enter a User ID before searching/i)).toBeTruthy();
});

test('handles search API error gracefully', async () => {
  searchFramesByImage.mockRejectedValue(new Error('Backend SigLIP2 model failed'));

  renderImageSearch({
    topK: 20,
    setTopK: jest.fn(),
    userId: 'team-a',
  });

  const fileInput = document.querySelector('input[type="file"]');
  const file = new File(['dummy-image'], 'test.png', { type: 'image/png' });
  fireEvent.change(fileInput, { target: { files: [file] } });

  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  await waitFor(() => {
    expect(screen.getByText('Backend SigLIP2 model failed')).toBeTruthy();
  });
});

test('New Search resets image file and clears results', async () => {
  searchFramesByImage.mockResolvedValue({
    results: [{
      frame_id: 'f1',
      video_id: 'V1',
      frame_idx: 1,
      timestamp_ms: 100,
      score: 0.9,
      frame_ids: ['f1'],
      timestamps_ms: [100],
      metadata: {},
    }],
    latency: { total_ms: 20 },
  });

  renderImageSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });

  const fileInput = document.querySelector('input[type="file"]');
  fireEvent.change(fileInput, { target: { files: [new File([''], 'test.png', { type: 'image/png' })] } });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  await waitFor(() => {
    expect(screen.getByText(/V1/)).toBeTruthy();
  });

  fireEvent.click(screen.getByRole('button', { name: 'New Search' }));

  expect(screen.queryByText(/V1/)).toBeNull();
  expect(screen.queryByText('test.png')).toBeNull();
  expect(screen.getByText(/Choose or drop an image/i)).toBeTruthy();
});
