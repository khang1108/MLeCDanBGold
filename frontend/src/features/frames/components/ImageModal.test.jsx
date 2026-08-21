import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ImageModal from './ImageModal';
import { getS3VideoUrl } from '../videoSource';

jest.mock('../videoSource', () => ({
  displayVideoId: (videoId) => videoId.split('.').at(-1),
  getS3VideoUrl: jest.fn(),
  timestampSeconds: (timestampMs) => (
    Number.isFinite(timestampMs) ? timestampMs / 1000 : null
  ),
}));

const frame = {
  frame_id: 'f1',
  video_id: 'L21_V001',
  frame_idx: 125,
  fps: 25,
  timestamp_ms: 5_000,
  caption: 'A frame caption',
  scores: { final: 0.9 },
};

beforeEach(() => getS3VideoUrl.mockResolvedValue('https://signed.example/video.mp4'));

test('loads the S3 video and seeks to the selected frame after metadata loads', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  expect(video.getAttribute('src')).toBe(
    'https://signed.example/video.mp4',
  );

  fireEvent.loadedMetadata(video);
  expect(video.currentTime).toBe(5);
  expect(screen.getByText('Time 5.000 s')).toBeTruthy();
});

test('prefers canonical timestamp over frame_idx/fps when they identify different moments', async () => {
  render(<ImageModal frame={{
    ...frame,
    frame_idx: 125,
    fps: 25,
    timestamp_ms: 5_200,
  }} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  fireEvent.loadedMetadata(video);

  expect(video.currentTime).toBe(5.2);
});

test('updates playback time without inventing a BTC frame index', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);
  const video = await screen.findByLabelText('Video for L21_V001');

  Object.defineProperty(video, 'currentTime', { configurable: true, value: 12.48 });
  fireEvent.timeUpdate(video);

  expect(screen.getByText('Time 12.480 s')).toBeTruthy();
  expect(screen.queryByText('Frame 312')).toBeNull();
});

test('shows a configuration message instead of requesting the full-size image when video playback is unavailable', async () => {
  getS3VideoUrl.mockResolvedValueOnce(null);
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  expect(await screen.findByText(/Video playback is unavailable/i)).toBeTruthy();
  expect(screen.queryByRole('img')).toBeNull();
});

test('requires canonical timestamp instead of deriving seek time from frame_idx', async () => {
  render(<ImageModal frame={{
    ...frame,
    frame_id: 'trake-frame',
    scores: undefined,
    timestamp_ms: undefined,
  }} onClose={jest.fn()} />);

  expect(await screen.findByText(/missing timestamp_ms/i)).toBeTruthy();
  expect(screen.queryByLabelText('Video for L21_V001')).toBeNull();
  expect(screen.queryByText('Retrieval Stage Scores')).toBeNull();
  expect(screen.queryByText('Final Relevance')).toBeNull();
  expect(screen.queryByText('Timestamp')).toBeNull();
});
