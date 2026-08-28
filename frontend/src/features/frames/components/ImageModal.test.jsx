import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ImageModal from './ImageModal';
import { getYouTubeEmbedUrl, getYouTubeWatchUrl } from '../videoSource';

jest.mock('../videoSource', () => ({
  displayVideoId: (videoId) => videoId.split('.').at(-1),
  getYouTubeEmbedUrl: jest.fn(),
  getYouTubeWatchUrl: jest.fn(),
  timestampSeconds: (timestampMs) => (
    Number.isFinite(timestampMs) ? timestampMs / 1000 : null
  ),
}));

jest.mock('./YouTubePlayer', () => ({
  __esModule: true,
  default: ({ embedUrl, title, targetTime, onTimeUpdate }) => (
    <>
      <iframe
        title={title}
        src={embedUrl}
        data-target-time={targetTime}
        tabIndex="0"
        onLoad={() => onTimeUpdate?.(targetTime)}
      />
      <button type="button" onClick={() => onTimeUpdate?.(5.2)}>Report player time</button>
    </>
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

beforeEach(() => {
  jest.clearAllMocks();
  getYouTubeEmbedUrl.mockReturnValue(
    'https://www.youtube.com/embed/Rzpw5WR7nAY?enablejsapi=1',
  );
  getYouTubeWatchUrl.mockReturnValue('https://www.youtube.com/watch?v=Rzpw5WR7nAY');
});

test('embeds the mapped YouTube video and passes the selected timestamp', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  const video = await screen.findByTitle('Video for L21_V001');
  expect(video.getAttribute('src')).toContain('/embed/Rzpw5WR7nAY');
  expect(video.getAttribute('data-target-time')).toBe('5');
  expect(screen.getByText('125')).toBeTruthy();
  expect(screen.getByText('5000 ms')).toBeTruthy();
  expect(screen.getByText('L21_V001 · 125')).toBeTruthy();
  expect(screen.queryByText(/BTC frame 125/i)).toBeNull();
  expect(screen.getByText('FPS')).toBeTruthy();
  expect(screen.getByText('25')).toBeTruthy();
});

test('shows the active query above the frame inspector without a label', () => {
  render(
    <ImageModal
      frame={frame}
      query={'a person enters the room E1 and sits down'}
      onClose={jest.fn()}
    />,
  );

  expect(screen.getByRole('status', { name: 'Current query' })).toBeTruthy();
  expect(screen.getByText('a person enters the room E1 and sits down')).toBeTruthy();
  expect(screen.queryByText('Current query')).toBeNull();
});

test('prefers canonical timestamp over frame_idx/fps when they identify different moments', async () => {
  render(<ImageModal frame={{ ...frame, timestamp_ms: 5_200 }} onClose={jest.fn()} />);

  const video = await screen.findByTitle('Video for L21_V001');
  expect(video.getAttribute('data-target-time')).toBe('5.2');
});

test('does not render a frame index overlay on the video', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  expect(await screen.findByTitle('Video for L21_V001')).toBeTruthy();
  expect(screen.queryByText(/keyframe/i)).toBeNull();
});

test('keeps the direct iframe focusable for native YouTube keyboard controls', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  const video = await screen.findByTitle('Video for L21_V001');
  expect(video.getAttribute('tabindex')).toBe('0');
});

test('uses the real player time for metadata while keeping the selected frame in the header', async () => {
  render(<ImageModal frame={{ ...frame, fps: 30 }} onClose={jest.fn()} />);

  fireEvent.click(await screen.findByRole('button', { name: 'Report player time' }));

  expect(screen.getByText('156')).toBeTruthy();
  expect(screen.getByText('5200 ms')).toBeTruthy();
  expect(screen.getByText('L21_V001 · 125')).toBeTruthy();
});

test('submits the live video position from the inspector header', async () => {
  const onSubmit = jest.fn();
  render(
    <ImageModal
      frame={{ ...frame, fps: 30 }}
      onSubmit={onSubmit}
      onClose={jest.fn()}
    />,
  );

  fireEvent.click(await screen.findByRole('button', { name: 'Report player time' }));
  fireEvent.click(screen.getByRole('button', { name: /submit current frame/i }));

  expect(onSubmit).toHaveBeenCalledWith({
    line: 'L21_V001,156',
    source: 'Frame inspector',
  });
});

test('shows an unavailable message when no YouTube metadata is mapped', async () => {
  getYouTubeEmbedUrl.mockReturnValueOnce(null);
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  expect(await screen.findByText(/Video playback is unavailable/i)).toBeTruthy();
  expect(screen.queryByRole('img')).toBeNull();
});

test('requires canonical timestamp instead of deriving seek time from frame_idx', async () => {
  render(<ImageModal frame={{ ...frame, scores: undefined, timestamp_ms: undefined }} onClose={jest.fn()} />);

  expect(await screen.findByText(/missing timestamp_ms/i)).toBeTruthy();
  expect(screen.queryByTitle('Video for L21_V001')).toBeNull();
  expect(screen.queryByText('Timestamp')).toBeNull();
});

test('closes the inspector when Escape is pressed', () => {
  const onClose = jest.fn();
  const { unmount } = render(<ImageModal frame={frame} onClose={onClose} />);

  fireEvent.keyDown(window, { key: 'Escape' });

  expect(onClose).toHaveBeenCalledTimes(1);
  unmount();
});
