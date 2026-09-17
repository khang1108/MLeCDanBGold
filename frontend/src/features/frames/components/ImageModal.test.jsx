import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import ImageModal from './ImageModal';
import { resolveFrameAtTimestamp } from '../../../api/frames';

jest.mock('../../../api/frames', () => ({
  resolveFrameAtTimestamp: jest.fn(),
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

  expect(screen.queryByText('Event Alignment')).toBeNull();
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

test('Step 1: shows Open EventTrail button under inspector when context exists without active state', () => {
  const onOpen = jest.fn();
  const context = {
    snapshotId: 'snap_1',
    resultId: 'r_1',
    kisRevision: 1,
    events: [{ id: 'E1', text: 'event 1' }],
  };
  render(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      eventTrail={{
        context,
        state: null,
        pending: false,
        open: onOpen,
      }}
    />,
  );

  const openBtn = screen.getByRole('button', { name: /open eventtrail/i });
  expect(openBtn).toBeTruthy();
  expect(openBtn.disabled).toBe(false);

  fireEvent.click(openBtn);
  expect(onOpen).toHaveBeenCalledWith(context);
});

test('Step 2: renders EventTrailPanel when state exists; selecting E2 + explore seeks player to candidate', async () => {
  const state = {
    session_id: 'ses_1',
    result_id: 'r_1',
    video_id: 'L21_V001',
    kis_revision: 1,
    trail_revision: 1,
    status: 'active',
    path: [
      { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 2000 },
      { event_id: 'E2', frame_id: 'f2', frame_idx: 20, timestamp_ms: 7000 },
    ],
    last_valid_path: null,
    approved_event_ids: [],
    rejected_counts: {},
    window: null,
    submission_selection: null,
    transition: null,
  };

  render(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [{ id: 'E1', text: 'e1' }, { id: 'E2', text: 'e2' }] },
        state,
        pending: false,
        act: jest.fn(),
      }}
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

  expect(screen.getByRole('region', { name: /eventtrail exploration/i })).toBeTruthy();
  // Select E2
  fireEvent.click(screen.getByTestId('event-rail-item-E2'));
  expect(screen.getAllByText('e2').length).toBeGreaterThanOrEqual(1);
});

test('Step 3: auto-seeks only on successful Decline replacement of selected event, not Approve or exhausted', async () => {
  let currentState = {
    session_id: 'ses_1',
    result_id: 'r_1',
    video_id: 'L21_V001',
    kis_revision: 1,
    trail_revision: 1,
    status: 'active',
    path: [
      { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 2000 },
      { event_id: 'E2', frame_id: 'f2', frame_idx: 20, timestamp_ms: 7000 },
    ],
    last_valid_path: null,
    approved_event_ids: [],
    rejected_counts: {},
    window: null,
    submission_selection: null,
    transition: null,
  };

  const actMock = jest.fn();

  const { rerender } = render(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [{ id: 'E1', text: 'e1' }, { id: 'E2', text: 'e2' }] },
        state: currentState,
        pending: false,
        act: actMock,
      }}
    />,
  );

  const video = await screen.findByLabelText('Video for L21_V001');
  let currentTime = 2;
  Object.defineProperty(video, 'duration', { configurable: true, value: 30 });
  Object.defineProperty(video, 'currentTime', {
    configurable: true,
    get: () => currentTime,
    set: (value) => { currentTime = value; },
  });
  fireEvent.loadedMetadata(video);

  // Select E2
  fireEvent.click(screen.getByTestId('event-rail-item-E2'));

  // Decline E2
  actMock.mockImplementation(async () => {
    currentState = {
      ...currentState,
      trail_revision: 2,
      path: [
        { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 2000 },
        { event_id: 'E2', frame_id: 'f2_new', frame_idx: 25, timestamp_ms: 9500 },
      ],
      transition: {
        action_event_id: 'E2',
        direct_changed_event_ids: ['E2'],
        indirect_changed_event_ids: [],
        candidate_diffs: [
          { event_id: 'E2', before_frame_id: 'f2', after_frame_id: 'f2_new', before_timestamp_ms: 7000, after_timestamp_ms: 9500 },
        ],
      },
    };
  });

  const declineBtn = screen.getByRole('button', { name: /^decline$/i });
  fireEvent.click(declineBtn);
  expect(actMock).toHaveBeenCalledWith({ type: 'decline', event_id: 'E2' });

  // Rerender with updated state
  rerender(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [{ id: 'E1', text: 'e1' }, { id: 'E2', text: 'e2' }] },
        state: currentState,
        pending: false,
        act: actMock,
      }}
    />
  );

  expect(currentTime).toBe(9.5);
});

test('Step 4: Use resolves canonical frame at player time and acts on EventTrail', async () => {
  resolveFrameAtTimestamp.mockResolvedValueOnce({
    frame_id: 'f12',
    video_id: 'L21_V001',
    requested_timestamp_ms: 12345,
    frame_idx: 300,
    timestamp_ms: 12000,
    metadata: {},
  });

  const actMock = jest.fn();
  const state = {
    session_id: 'ses_1',
    result_id: 'r_1',
    video_id: 'L21_V001',
    kis_revision: 1,
    trail_revision: 1,
    status: 'active',
    path: [
      { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 2000 },
      { event_id: 'E2', frame_id: 'f2', frame_idx: 20, timestamp_ms: 7000 },
    ],
    last_valid_path: null,
    approved_event_ids: [],
    rejected_counts: {},
    window: null,
    submission_selection: null,
    transition: null,
  };

  render(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [{ id: 'E1', text: 'e1' }, { id: 'E2', text: 'e2' }] },
        state,
        pending: false,
        act: actMock,
      }}
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
  currentTime = 12.345;

  // Select E2
  fireEvent.click(screen.getByTestId('event-rail-item-E2'));

  // Click Use
  const useBtn = screen.getByRole('button', { name: /^use$/i });
  fireEvent.click(useBtn);

  await waitFor(() => {
    expect(resolveFrameAtTimestamp).toHaveBeenCalledWith({
      videoId: 'L21_V001',
      timestampMs: 12345,
    });
    expect(actMock).toHaveBeenCalledWith({
      type: 'use_frame',
      event_id: 'E2',
      frame_id: 'f12',
    });
  });
});

test('Step 5: Trail Submit uses submission_selection and hides header submit button', async () => {
  const onOpenSubmission = jest.fn();
  const state = {
    session_id: 'ses_1',
    result_id: 'r_1',
    video_id: 'L21_V001',
    kis_revision: 1,
    trail_revision: 2,
    status: 'active',
    path: [
      { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 2000 },
      { event_id: 'E2', frame_id: 'f12', frame_idx: 300, timestamp_ms: 12000 },
    ],
    last_valid_path: null,
    approved_event_ids: ['E2'],
    rejected_counts: {},
    window: null,
    submission_selection: {
      event_id: 'E2',
      frame_id: 'f12',
      frame_idx: 300,
      timestamp_ms: 12000,
    },
    transition: null,
  };

  const { rerender } = render(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      onOpenSubmission={onOpenSubmission}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [{ id: 'E1', text: 'e1' }, { id: 'E2', text: 'e2' }] },
        state: { ...state, submission_selection: null },
        pending: false,
        act: jest.fn(),
      }}
    />,
  );

  // Header submit button is hidden while Trail is active
  expect(screen.queryByRole('button', { name: 'Submit current video moment to DRES' })).toBeNull();

  // Trail submit is disabled without submission_selection and displays hint
  const submitBtn = screen.getByRole('button', { name: /^submit$/i });
  expect(submitBtn.disabled).toBe(true);
  expect(screen.getByText('Use a frame before submitting from EventTrail.')).toBeTruthy();

  // Rerender with submission_selection
  rerender(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      onOpenSubmission={onOpenSubmission}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [{ id: 'E1', text: 'e1' }, { id: 'E2', text: 'e2' }] },
        state,
        pending: false,
        act: jest.fn(),
      }}
    />,
  );

  expect(submitBtn.disabled).toBe(false);
  fireEvent.click(submitBtn);

  expect(onOpenSubmission).toHaveBeenCalledWith({
    videoId: 'L21_V001',
    startMs: 12000,
    endMs: 12000,
  });
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

test('Exit EventTrail button calls eventTrail.close', () => {
  const closeMock = jest.fn();
  const state = {
    session_id: 'ses_1',
    result_id: 'r_1',
    video_id: 'L21_V001',
    kis_revision: 1,
    trail_revision: 2,
    status: 'active',
    path: [],
    last_valid_path: null,
    approved_event_ids: [],
    rejected_counts: {},
    window: null,
    submission_selection: null,
    transition: null,
  };

  render(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [] },
        state,
        pending: false,
        close: closeMock,
      }}
    />,
  );

  const exitButtons = screen.getAllByRole('button', { name: /^exit$/i });
  expect(exitButtons.length).toBeGreaterThanOrEqual(1);

  fireEvent.click(exitButtons[0]);
  expect(closeMock).toHaveBeenCalledWith({ suppressError: true });
});


