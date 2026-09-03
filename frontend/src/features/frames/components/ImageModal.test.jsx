import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ImageModal from './ImageModal';

const frame = {
  frame_id: 'f1',
  video_id: 'L21_V001',
  frame_idx: 125,
  fps: 25,
  timestamp_ms: 5_000,
  caption: 'A frame caption',
  scores: { final: 0.9 },
};

test('streams the canonical video at the selected timestamp', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  expect(video.tagName).toBe('VIDEO');
  expect(video.getAttribute('src')).toBe(
    'https://stream.iamphuckhang.dev/api/v1/videos/L21_V001/stream',
  );
  expect(video.hasAttribute('controls')).toBe(false);
  expect(screen.getByRole('slider', { name: 'Video timeline' })).toBeTruthy();
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

test('updates the stream URL when the selected timestamp changes', async () => {
  const { rerender } = render(<ImageModal frame={frame} onClose={jest.fn()} />);

  rerender(<ImageModal frame={{ ...frame, timestamp_ms: 5_200 }} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  expect(video.getAttribute('src')).toBe(
    'https://stream.iamphuckhang.dev/api/v1/videos/L21_V001/stream',
  );
});

test('seeks the raw stream to the selected source timestamp after metadata loads', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  let currentTime = 0;
  Object.defineProperty(video, 'duration', { configurable: true, value: 30 });
  Object.defineProperty(video, 'currentTime', {
    configurable: true,
    get: () => currentTime,
    set: (value) => { currentTime = value; },
  });

  fireEvent.loadedMetadata(video);

  expect(currentTime).toBe(5);
  expect(screen.getByText('5000 ms')).toBeTruthy();
});

test('keeps metadata on playback time while hover preview stays non-seeking', async () => {
  render(<ImageModal frame={{ ...frame, timestamp_ms: 2_000 }} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  let currentTime = 0;
  Object.defineProperty(video, 'duration', { configurable: true, value: 10 });
  Object.defineProperty(video, 'currentTime', {
    configurable: true,
    get: () => currentTime,
    set: (value) => { currentTime = value; },
  });
  fireEvent.loadedMetadata(video);

  const timeline = screen.getByTestId('video-timeline-track');
  Object.defineProperty(timeline, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({ left: 0, width: 200, top: 0, right: 200, bottom: 10, height: 10 }),
  });
  fireEvent.mouseMove(timeline, { clientX: 160 });

  expect(currentTime).toBe(2);
  expect(screen.getByText('2000 ms')).toBeTruthy();

  fireEvent.change(screen.getByRole('slider', { name: 'Video timeline' }), {
    target: { value: '8' },
  });
  expect(currentTime).toBe(8);
  expect(screen.getByText('8000 ms')).toBeTruthy();
});

test('does not render a frame index overlay on the video', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  expect(await screen.findByLabelText('Video for L21_V001')).toBeTruthy();
  expect(screen.queryByText(/keyframe/i)).toBeNull();
});

test('renders custom stream controls with a hover-preview timeline', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  expect(video.hasAttribute('controls')).toBe(false);
  expect(screen.getByRole('button', { name: 'Play video' })).toBeTruthy();
  expect(screen.getByRole('slider', { name: 'Video timeline' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Unmute video' })).toBeTruthy();
  expect(screen.getByRole('slider', { name: 'Video volume' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Enter fullscreen' })).toBeTruthy();
  expect(screen.queryByText('Hover timeline for frame preview')).toBeNull();
});

test('toggles playback with Space when the inspector has focus', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  video.play = jest.fn(() => Promise.resolve());
  const modalCard = document.querySelector('.modal-card');
  fireEvent.keyDown(modalCard, { key: ' ' });

  expect(video.play).toHaveBeenCalledTimes(1);
});

test('starts progressive playback automatically without waiting for the full file', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  expect(video.autoplay).toBe(true);
  expect(video.muted).toBe(true);
  expect(video.preload).toBe('metadata');
});

test('uses source time for metadata while keeping the selected frame in the header', async () => {
  render(<ImageModal frame={{ ...frame, fps: 30 }} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  Object.defineProperty(video, 'currentTime', { configurable: true, value: 5.2 });
  fireEvent.timeUpdate(video);

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

  const video = await screen.findByLabelText('Video for L21_V001');
  Object.defineProperty(video, 'currentTime', { configurable: true, value: 5.2 });
  fireEvent.timeUpdate(video);
  fireEvent.click(screen.getByRole('button', { name: /submit current frame/i }));

  expect(onSubmit).toHaveBeenCalledWith({
    line: 'L21_V001,156',
    source: 'Frame inspector',
  });
});

test('uses normalized fps when calculating the live BTC frame index', async () => {
  const onSubmit = jest.fn();
  render(
    <ImageModal
      frame={{ ...frame, fps: 29.97 }}
      onSubmit={onSubmit}
      onClose={jest.fn()}
    />,
  );

  const video = await screen.findByLabelText('Video for L21_V001');
  Object.defineProperty(video, 'currentTime', { configurable: true, value: 5.25 });
  fireEvent.timeUpdate(video);

  expect(screen.getByText('158')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: /submit current frame/i }));

  expect(onSubmit).toHaveBeenCalledWith({
    line: 'L21_V001,158',
    source: 'Frame inspector',
  });
});

test('shows an unavailable message when the stream cannot be built', async () => {
  render(<ImageModal frame={{ ...frame, video_id: '' }} onClose={jest.fn()} />);

  expect(await screen.findByText(/Video playback is unavailable/i)).toBeTruthy();
  expect(screen.queryByRole('img')).toBeNull();
});

test('requires canonical timestamp instead of deriving seek time from frame_idx', async () => {
  render(<ImageModal frame={{ ...frame, scores: undefined, timestamp_ms: undefined }} onClose={jest.fn()} />);

  expect(await screen.findByText(/missing timestamp_ms/i)).toBeTruthy();
  expect(screen.queryByTitle('Video for L21_V001')).toBeNull();
  expect(screen.queryByText('Timestamp')).toBeNull();
});

test('supports manual video inspection without inventing frame identity', async () => {
  render(<ImageModal frame={{ video_id: 'V01', timestamp_ms: 12_000 }} onClose={jest.fn()} />);

  expect(await screen.findByLabelText('Video for V01')).toBeTruthy();
  expect(screen.getByText('V01 · 12000 ms')).toBeTruthy();
  expect(screen.queryByText('Internal frame ID')).toBeNull();
  expect(screen.queryByText('BTC frame index')).toBeNull();
  expect(screen.queryByRole('button', { name: /submit current frame/i })).toBeNull();
});

test('closes the inspector when Escape is pressed', () => {
  const onClose = jest.fn();
  const { unmount } = render(<ImageModal frame={frame} onClose={onClose} />);

  fireEvent.keyDown(window, { key: 'Escape' });

  expect(onClose).toHaveBeenCalledTimes(1);
  unmount();
});
