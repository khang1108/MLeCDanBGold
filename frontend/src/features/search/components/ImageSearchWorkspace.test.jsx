import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFramesByImage } from '../../../api/search';
import ImageSearchWorkspace from './ImageSearchWorkspace';
import {
  createQueryHistory,
  markFrameViewed,
} from '../../../api/history';

jest.mock('../../../api/search');
jest.mock('../../../api/history', () => ({
  createQueryHistory: jest.fn(),
  markFrameViewed: jest.fn(),
}));

const renderImageSearch = (props) => render(<ImageSearchWorkspace {...props} />);

beforeEach(() => {
  searchFramesByImage.mockReset();
  createQueryHistory.mockResolvedValue({});
  markFrameViewed.mockResolvedValue({});
  global.URL.createObjectURL = jest.fn(() => 'blob:mock-image-preview');
  global.URL.revokeObjectURL = jest.fn();
});

afterEach(() => {
  delete global.URL.createObjectURL;
  delete global.URL.revokeObjectURL;
});

test('renders empty image dropzone with disabled Search button', () => {
  renderImageSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });

  expect(screen.getByText(/Choose, drop, or paste an image/i)).toBeTruthy();
  const searchBtn = screen.getByRole('button', { name: 'Search' });
  expect(searchBtn.disabled).toBe(true);
});

test('does not render a shared answer workspace in Image Search', () => {
  renderImageSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });

  expect(screen.queryByRole('region', { name: /answer workspace/i })).toBeNull();
});

test('pasting an image via clipboard (Ctrl+V) selects image and enables Search', () => {
  renderImageSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });

  const pastedFile = new File(['pasted-data'], 'clipboard-photo.png', { type: 'image/png' });
  const clipboardEvent = new Event('paste', { bubbles: true, cancelable: true });
  clipboardEvent.clipboardData = {
    items: [
      {
        type: 'image/png',
        getAsFile: () => pastedFile,
      },
    ],
  };

  fireEvent(window, clipboardEvent);

  expect(screen.getByText('clipboard-photo.png')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Search' }).disabled).toBe(false);
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

  const onFrameClick = jest.fn();

  renderImageSearch({
    topK: 15,
    setTopK: jest.fn(),
    userId: 'team-a',
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
      userId: 'team-a',
    });
  });

  await waitFor(() => {
    expect(document.querySelector('.latency-summary')?.textContent).toMatch(/Found\s+1\s+frames in\s+47ms/);
  });

  expect(createQueryHistory).not.toHaveBeenCalled();
  expect(markFrameViewed).not.toHaveBeenCalled();

  // Click frame
  const frameCard = screen.getByText(/L01_V001/).closest('.frame-card');
  fireEvent.click(frameCard);
  expect(onFrameClick).toHaveBeenCalledWith({
    frame: mockResults[0],
  });
});

test('opens direct submission for an image-search frame at its exact timestamp', async () => {
  const onOpenSubmission = jest.fn();
  searchFramesByImage.mockResolvedValueOnce({
    results: [{
      frame_id: 'image-time-frame',
      video_id: 'V02',
      frame_idx: 5,
      timestamp_ms: 9_876,
      frame_ids: ['image-time-frame'],
      timestamps_ms: [9_876],
      metadata: {},
    }],
    latency: { total_ms: 2 },
  });
  renderImageSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a', onOpenSubmission });
  fireEvent.change(document.querySelector('input[type="file"]'), {
    target: { files: [new File(['image'], 'query.png', { type: 'image/png' })] },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  fireEvent.click(await screen.findByRole('button', { name: 'Submit this frame to DRES' }));
  expect(onOpenSubmission).toHaveBeenCalledWith({ videoId: 'V02', startMs: 9_876, endMs: 9_876 });
});

test('searches without a User ID and does not create history', async () => {
  searchFramesByImage.mockResolvedValue({
    results: [],
    latency: { total_ms: 20 },
  });

  renderImageSearch({
    topK: 20,
    setTopK: jest.fn(),
    userId: '',
  });

  const fileInput = document.querySelector('input[type="file"]');
  const file = new File(['dummy-image'], 'test.png', { type: 'image/png' });
  fireEvent.change(fileInput, { target: { files: [file] } });

  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  await waitFor(() => {
    expect(searchFramesByImage).toHaveBeenCalledWith({
      imageFile: file,
      topK: 20,
      signal: expect.any(AbortSignal),
      userId: '',
    });
  });

  expect(createQueryHistory).not.toHaveBeenCalled();
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
  expect(screen.getByText(/Choose, drop, or paste an image/i)).toBeTruthy();
});
