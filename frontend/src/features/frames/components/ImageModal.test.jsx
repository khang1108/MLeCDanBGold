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

test('submits current video_id and playback timestamp when submit button is clicked', async () => {
  const onOpenSubmission = jest.fn();
  render(
    <ImageModal
      frame={{ ...frame, fps: 30 }}
      onOpenSubmission={onOpenSubmission}
      onClose={jest.fn()}
    />,
  );

  const video = await screen.findByLabelText('Video for L21_V001');
  Object.defineProperty(video, 'currentTime', { configurable: true, value: 5.2 });
  fireEvent.timeUpdate(video);

  const submitButtons = screen.getAllByRole('button', { name: /submit/i });
  expect(submitButtons.length).toBeGreaterThan(0);
  fireEvent.click(submitButtons[0]);

  expect(onOpenSubmission).toHaveBeenCalledWith({
    videoId: 'L21_V001',
    startMs: 5200,
    endMs: 5200,
  });
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



test('Step 1: does not show Open EventTrail button under inspector when clicking a frame', () => {
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
        open: jest.fn(),
      }}
    />,
  );

  expect(screen.queryByRole('button', { name: /open eventtrail/i })).toBeNull();
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

  expect(screen.getByRole('region', { name: /hypothesis explorer|eventtrail exploration/i })).toBeTruthy();
  // Select E2
  fireEvent.click(screen.getByTestId('event-rail-item-E2'));
  expect(screen.getAllByText('e2').length).toBeGreaterThanOrEqual(1);
});

test('Result Hypothesis Explorer wires focus, preview, Keep, Use, and Reject actions end-to-end', async () => {
  const focusEvent = jest.fn();
  const previewAlternative = jest.fn();
  const clearPreview = jest.fn();
  const keep = jest.fn();
  const useAlternative = jest.fn();
  const rejectMode = jest.fn();
  const alternative = {
    alternative_id: 'opaque-alternative-42',
    event_id: 'E2',
    is_current: true,
    path: [
      { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 2000 },
      { event_id: 'E2', frame_id: 'f2b', frame_idx: 21, timestamp_ms: 7200 },
    ],
  };
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

  const { rerender } = render(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [{ id: 'E1', text: 'e1' }, { id: 'E2', text: 'e2' }] },
        state,
        pending: false,
        alternatives: [alternative],
        isLoadingAlternatives: false,
        focusedEventId: 'E2',
        currentQueryRevision: 1,
        focusEvent,
        previewAlternative,
        clearPreview,
        keep,
        useAlternative,
        rejectMode,
      }}
    />,
  );

  fireEvent.click(screen.getByTestId('event-rail-item-E2'));
  expect(focusEvent).toHaveBeenCalledWith('E2');

  fireEvent.click(screen.getByRole('button', { name: /preview alternative/i }));
  expect(previewAlternative).toHaveBeenCalledWith(alternative);
  expect(keep).not.toHaveBeenCalled();
  expect(useAlternative).not.toHaveBeenCalled();
  expect(rejectMode).not.toHaveBeenCalled();

  rerender(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [{ id: 'E1', text: 'e1' }, { id: 'E2', text: 'e2' }] },
        state,
        pending: false,
        alternatives: [alternative],
        previewAlternativeState: alternative,
        isLoadingAlternatives: false,
        focusedEventId: 'E2',
        currentQueryRevision: 1,
        focusEvent,
        previewAlternative,
        clearPreview,
        keep,
        useAlternative,
        rejectMode,
      }}
    />,
  );

  fireEvent.click(screen.getByRole('button', { name: /^keep$/i }));
  expect(keep).toHaveBeenCalledWith('E2');

  fireEvent.click(screen.getAllByRole('button', { name: /use this occurrence/i })[0]);
  expect(useAlternative).toHaveBeenCalledWith('E2', 'opaque-alternative-42');

  fireEvent.click(screen.getByRole('button', { name: /reject occurrence/i }));
  expect(rejectMode).toHaveBeenCalledWith('E2', 'opaque-alternative-42');
  expect(clearPreview).not.toHaveBeenCalled();
});

test('Step 3: Result Hypothesis Explorer Reject is disabled without an opaque alternative id', async () => {
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

  const rejectMode = jest.fn();

  render(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [{ id: 'E1', text: 'e1' }, { id: 'E2', text: 'e2' }] },
        state,
        pending: false,
        focusedEventId: 'E2',
        rejectMode,
      }}
    />,
  );

  fireEvent.click(screen.getByTestId('event-rail-item-E2'));

  const rejectBtn = screen.getByRole('button', { name: /reject occurrence/i });
  expect(rejectBtn).toBeDisabled();
  fireEvent.click(rejectBtn);
  expect(rejectMode).not.toHaveBeenCalled();
});

test('Step 4: Use current video timestamp replaces canonical candidate and sets submission_selection', async () => {
  const actMock = jest.fn();
  resolveFrameAtTimestamp.mockResolvedValueOnce({
    video_id: 'L21_V001',
    frame_id: 'f12',
    frame_idx: 120,
    timestamp_ms: 12345,
  });

  render(
    <ImageModal
      frame={frame}
      onClose={jest.fn()}
      eventTrail={{
        context: { snapshotId: 'snap_1', resultId: 'r_1', kisRevision: 1, events: [{ id: 'E1', text: 'e1' }, { id: 'E2', text: 'e2' }] },
        state: {
          session_id: 'ses_1',
          result_id: 'r_1',
          video_id: 'L21_V001',
          kis_revision: 1,
          trail_revision: 0,
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
        },
        pending: false,
        act: actMock,
      }}
    />
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
  const useBtn = screen.getByRole('button', { name: /use \(manual frame\)|^use$/i });
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
  expect(screen.getByText('Use a frame before submitting from Hypothesis Explorer.')).toBeTruthy();

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

test('does not render bottom alignment section in ImageModal', () => {
  const alignedFrame = {
    ...frame,
    frame_ids: ['f1', 'f2'],
    timestamps_ms: [1200, 2400],
  };
  render(<ImageModal frame={alignedFrame} events={['hold', 'roll']} onClose={jest.fn()} />);
  expect(document.querySelector('.modal-bottom-alignment-section')).toBeNull();
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

test('renders Select candidate button in AVS mode and toggles selection on click', () => {
  const onToggleMock = jest.fn();
  const { rerender } = render(
    <ImageModal
      frame={{ frame_id: 'f_avs', video_id: 'L21_V001', timestamp_ms: 5000 }}
      onClose={jest.fn()}
      isAvsMode={true}
      isCandidateSelected={false}
      onToggleCandidateSelection={onToggleMock}
    />
  );

  const selectBtns = screen.getAllByRole('button', { name: /select candidate/i });
  expect(selectBtns.length).toBeGreaterThanOrEqual(1);
  expect(selectBtns[0].textContent).toContain('Select');

  fireEvent.click(selectBtns[0]);
  expect(onToggleMock).toHaveBeenCalledTimes(1);

  // Rerender as selected
  rerender(
    <ImageModal
      frame={{ frame_id: 'f_avs', video_id: 'L21_V001', timestamp_ms: 5000 }}
      onClose={jest.fn()}
      isAvsMode={true}
      isCandidateSelected={true}
      onToggleCandidateSelection={onToggleMock}
    />
  );

  const deselectBtns = screen.getAllByRole('button', { name: /deselect candidate/i });
  expect(deselectBtns.length).toBeGreaterThanOrEqual(1);
  expect(deselectBtns[0].textContent).toContain('Selected');
});

test('toggles theater / expanded player mode on button click and keyboard shortcut T', () => {
  render(
    <ImageModal
      frame={{ frame_id: 'f_test', video_id: 'L21_V001', timestamp_ms: 1000 }}
      onClose={jest.fn()}
    />
  );

  const modalCard = document.querySelector('.modal-card');
  expect(modalCard.classList.contains('is-theater-mode')).toBe(false);

  // Click Expand button
  const expandBtn = screen.getAllByRole('button', { name: /expand player/i })[0];
  fireEvent.click(expandBtn);
  expect(modalCard.classList.contains('is-theater-mode')).toBe(true);

  // Re-toggle via 't' key
  fireEvent.keyDown(modalCard, { key: 't' });
  expect(modalCard.classList.contains('is-theater-mode')).toBe(false);
});

test('toggles fit mode between contain and cover (fill) on button click and key C', () => {
  render(
    <ImageModal
      frame={{ frame_id: 'f_test', video_id: 'L21_V001', timestamp_ms: 1000 }}
      onClose={jest.fn()}
    />
  );

  const viewerCol = document.querySelector('.modal-viewer-column');
  expect(viewerCol.classList.contains('fit-contain')).toBe(true);

  // Click Fill frame button
  const fillBtn = screen.getAllByRole('button', { name: /fill frame/i })[0];
  fireEvent.click(fillBtn);
  expect(viewerCol.classList.contains('fit-cover')).toBe(true);

  // Toggle back with key 'c'
  const modalCard = document.querySelector('.modal-card');
  fireEvent.keyDown(modalCard, { key: 'c' });
  expect(viewerCol.classList.contains('fit-contain')).toBe(true);
});

test('renders Hypothesis Explorer button in header when eventTrail context is present and state is inactive, and clicking it calls open', () => {
  const openTrail = jest.fn();
  const eventTrailProp = {
    context: {
      snapshotId: 'snap_1',
      resultId: 'r_1',
      kisRevision: 1,
    },
    state: null,
    pending: false,
    open: openTrail,
  };

  render(
    <ImageModal
      frame={{ frame_id: 'f_test', video_id: 'L21_V001', timestamp_ms: 1000 }}
      eventTrail={eventTrailProp}
      onClose={jest.fn()}
    />
  );

  const explorerBtn = screen.getByRole('button', { name: /hypothesis explorer/i });
  expect(explorerBtn).toBeTruthy();

  fireEvent.click(explorerBtn);
  expect(openTrail).toHaveBeenCalledTimes(1);
});

test('renders trail error banner in Frame Inspector mode when eventTrail.error is set', () => {
  const eventTrailProp = {
    context: {
      snapshotId: 'snap_1',
      resultId: 'r_1',
      kisRevision: 1,
    },
    state: null,
    pending: false,
    error: 'Snapshot expired. Please rerun search.',
    open: jest.fn(),
  };

  render(
    <ImageModal
      frame={{ frame_id: 'f_test', video_id: 'L21_V001', timestamp_ms: 1000 }}
      eventTrail={eventTrailProp}
      onClose={jest.fn()}
    />
  );

  expect(screen.getByRole('alert')).toBeTruthy();
  expect(screen.getByText('Snapshot expired. Please rerun search.')).toBeTruthy();
});
