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
  expect(video.getAttribute('src')).toMatch(/\/videos\/L21_V001\/stream$/);
  expect(video.hasAttribute('controls')).toBe(false);
  expect(screen.getByRole('slider', { name: 'Video timeline' })).toBeTruthy();
  expect(screen.getByText('Frame Inspector')).toBeTruthy();
  expect(screen.getByText('L21_V001')).toBeTruthy();
  expect(screen.getByText('5000 ms')).toBeTruthy();
  expect(screen.queryByText('Video Controls')).toBeNull();
  expect(screen.queryByRole('button', { name: /submit/i })).toBeNull();
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
  expect(video.getAttribute('src')).toMatch(/\/videos\/L21_V001\/stream$/);
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

test('keeps canonical metadata while seeking manual inspection to the requested time', async () => {
  render(
    <ImageModal
      frame={{ ...frame, timestamp_ms: 5_000, metadata: { caption: 'Canonical evidence' } }}
      initialTimestampMs={12_000}
      onClose={jest.fn()}
    />,
  );

  const video = await screen.findByLabelText('Video for L21_V001');
  let currentTime = 0;
  Object.defineProperty(video, 'duration', { configurable: true, value: 30 });
  Object.defineProperty(video, 'currentTime', {
    configurable: true,
    get: () => currentTime,
    set: (value) => { currentTime = value; },
  });

  fireEvent.loadedMetadata(video);

  expect(currentTime).toBe(12);
  expect(screen.getByText('Frame Inspector')).toBeTruthy();
  expect(screen.getByText('L21_V001')).toBeTruthy();
  expect(screen.getByText('12000 ms')).toBeTruthy();
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

test('uses source time for metadata while keeping Frame Inspector in the header', async () => {
  render(<ImageModal frame={{ ...frame, fps: 30 }} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  Object.defineProperty(video, 'currentTime', { configurable: true, value: 5.2 });
  fireEvent.timeUpdate(video);

  expect(screen.getByText('5200 ms')).toBeTruthy();
  expect(screen.getByText('Frame Inspector')).toBeTruthy();
  expect(screen.getByText('L21_V001')).toBeTruthy();
});

test('does not expose a submit button even when onOpenSubmission is passed', async () => {
  render(
    <ImageModal
      frame={{ ...frame, fps: 30 }}
      onOpenSubmission={jest.fn()}
      onClose={jest.fn()}
    />,
  );

  const video = await screen.findByLabelText('Video for L21_V001');
  Object.defineProperty(video, 'currentTime', { configurable: true, value: 5.2 });
  fireEvent.timeUpdate(video);
  expect(screen.queryByRole('button', { name: /submit/i })).toBeNull();
});

test('does not render redundant video controls shortcuts card', () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  expect(screen.queryByText('Video Controls')).toBeNull();
  expect(screen.queryByText(/Play \/ Pause/i)).toBeNull();
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
  expect(screen.getByText('Frame Inspector')).toBeTruthy();
  expect(screen.getByText('V01')).toBeTruthy();
  expect(screen.getByText('12000 ms')).toBeTruthy();
  expect(screen.queryByText('Internal frame ID')).toBeNull();
  expect(screen.queryByText('BTC frame index')).toBeNull();
  expect(screen.queryByRole('button', { name: /submit current frame/i })).toBeNull();
});

test('shows keyframe fallback preview when video stream errors', async () => {
  render(<ImageModal frame={frame} onClose={jest.fn()} />);

  const video = await screen.findByLabelText('Video for L21_V001');
  fireEvent.error(video);

  expect(await screen.findByText(/Showing keyframe preview/i)).toBeTruthy();
  expect(screen.getByRole('img', { name: `Frame ${frame.frame_id}` })).toBeTruthy();
});

test('closes the inspector when Escape is pressed', () => {
  const onClose = jest.fn();
  const { unmount } = render(<ImageModal frame={frame} onClose={onClose} />);

  fireEvent.keyDown(window, { key: 'Escape' });

  expect(onClose).toHaveBeenCalledTimes(1);
  unmount();
});

test('renders alignment sequence in order and immediately open upon display', () => {
  const alignedFrame = {
    ...frame,
    frame_ids: ['f1', 'f2'],
    timestamps_ms: [1200, 2400],
  };
  render(
    <ImageModal
      frame={alignedFrame}
      events={['hold', 'roll']}
      onClose={jest.fn()}
    />,
  );

  expect(screen.getByText('Event Alignment')).toBeTruthy();
  expect(screen.getByText('2 events')).toBeTruthy();
  expect(screen.getByText('E1')).toBeTruthy();
  expect(screen.getByText('hold')).toBeTruthy();
  expect(screen.getByText('00:01.200')).toBeTruthy();
  expect(screen.getByText('E2')).toBeTruthy();
  expect(screen.getByText('roll')).toBeTruthy();
  expect(screen.getByText('00:02.400')).toBeTruthy();
});

test('seeking via alignment sequence row updates the video time', async () => {
  const alignedFrame = {
    ...frame,
    frame_ids: ['f1', 'f2'],
    timestamps_ms: [1200, 2400],
  };
  render(
    <ImageModal
      frame={alignedFrame}
      events={['hold', 'roll']}
      onClose={jest.fn()}
    />,
  );

  const seekButton = screen.getByRole('button', { name: '00:02.400' });
  fireEvent.click(seekButton);

  const video = await screen.findByLabelText('Video for L21_V001');
  expect(video.currentTime).toBe(2.4);
});

test('renders alignment section at modal bottom when events are present, omits when absent', () => {
  const { rerender } = render(<ImageModal frame={frame} onClose={jest.fn()} />);
  expect(document.querySelector('.modal-bottom-alignment-section')).toBeNull();

  const alignedFrame = {
    ...frame,
    frame_ids: ['f1', 'f2'],
    timestamps_ms: [1200, 2400],
  };
  rerender(<ImageModal frame={alignedFrame} events={['hold', 'roll']} onClose={jest.fn()} />);
  const bottomSection = document.querySelector('.modal-bottom-alignment-section');
  expect(bottomSection).toBeTruthy();
  expect(bottomSection.querySelector('.alignment-accordion.always-open')).toBeTruthy();
});

