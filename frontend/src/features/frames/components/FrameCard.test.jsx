import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import FrameCard from './FrameCard';

test('does not expose submission while the participant is disconnected', () => {
  const frame = {
    frame_id: 'internal-frame-1',
    video_id: 'L21_V001',
    frame_idx: 17794,
    caption: 'A sample frame',
  };
  const onOpenSubmission = jest.fn();
  render(<FrameCard frame={frame} onClick={jest.fn()} />);

  expect(screen.queryByRole('button', { name: /submit this frame/i })).toBeNull();
  expect(onOpenSubmission).not.toHaveBeenCalled();
});

test('shows the video id and timestamp without embedding alignment accordion inside the card', () => {
  const frame = {
    frame_id: 'representative-frame',
    video_id: 'L21_V001',
    frame_idx: 17794,
    score: 2.34567,
    timestamp_ms: 1200,
    frame_ids: ['f1', 'f2'],
    timestamps_ms: [1200, 2400],
    metadata: { caption: 'A sample frame' },
  };
  render(
    <FrameCard
      frame={frame}
      events={['hold', 'roll']}
      onClick={jest.fn()}
    />,
  );

  expect(screen.getByText('L21_V001')).toBeTruthy();
  expect(screen.getByText('1200 ms')).toBeTruthy();
  expect(screen.queryByRole('button', { name: /alignment/i })).toBeNull();
});

test('shows a loading placeholder while page details are fetched', () => {
  render(
    <FrameCard
      frame={{
        frame_id: 'page-frame',
        video_id: 'L21_a_topic.video-1',
        frame_idx: 1,
        timestamp_ms: 100,
      }}
      detailStatus="loading"
    />,
  );

  expect(screen.getByText('Loading frame…')).toBeTruthy();
});

test('does not add answer-derived styling to a card', () => {
  const frame = {
    frame_id: 'internal-frame-1',
    video_id: 'L21_V001',
    frame_idx: 17794,
    caption: 'A sample frame',
  };
  const { container } = render(<FrameCard frame={frame} className="viewed" onClick={jest.fn()} />);

  expect(container.querySelector('.frame-card').className).toBe('frame-card viewed');
});

test('opens direct submission with the canonical frame time and stops card propagation', () => {
  const frame = {
    frame_id: 'internal-frame-12000',
    video_id: 'V01',
    frame_idx: 4,
    timestamp_ms: 12_000,
  };
  const onOpenSubmission = jest.fn();
  const onClick = jest.fn();
  render(
    <FrameCard
      frame={frame}
      onOpenSubmission={onOpenSubmission}
      onClick={onClick}
    />,
  );

  fireEvent.click(screen.getByRole('button', { name: 'Submit this frame to DRES' }));

  expect(onOpenSubmission).toHaveBeenCalledWith({ videoId: 'V01', startMs: 12_000, endMs: 12_000 });
  expect(onClick).not.toHaveBeenCalled();
});

test('disables the frame action while the active DRES task is loading', () => {
  render(
    <FrameCard
      frame={{ frame_id: 'f1', video_id: 'V01', timestamp_ms: 500 }}
      isSubmissionOpening
      onOpenSubmission={jest.fn()}
    />,
  );

  expect(screen.getByRole('button', { name: 'Submit this frame to DRES' }).disabled).toBe(true);
});

test('freezes the canonical video and timestamp when display details differ', () => {
  const frame = { frame_id: 'canonical-frame', video_id: 'V01', frame_idx: 3, timestamp_ms: 12_345 };
  const onOpenSubmission = jest.fn();
  render(
    <FrameCard
      frame={frame}
      detail={{ video_id: 'V99', timestamp_ms: 99_000 }}
      onOpenSubmission={onOpenSubmission}
    />,
  );

  fireEvent.click(screen.getByRole('button', { name: 'Submit this frame to DRES' }));
  expect(onOpenSubmission).toHaveBeenCalledWith({ videoId: 'V01', startMs: 12_345, endMs: 12_345 });
});

test('Step 2 (Task 7): renders non-ranking badge when explored or exhausted without hiding or altering metadata', () => {
  const frame = { frame_id: 'f1', video_id: 'V01', timestamp_ms: 5000 };

  const { rerender } = render(<FrameCard frame={frame} annotation="explored" />);
  expect(screen.getByText('Explored')).toBeTruthy();

  rerender(<FrameCard frame={frame} annotation="exhausted" />);
  expect(screen.getByText('Exhausted')).toBeTruthy();

  rerender(<FrameCard frame={frame} annotation="unvisited" />);
  expect(screen.queryByText('Explored')).toBeNull();
  expect(screen.queryByText('Exhausted')).toBeNull();
});

test('renders eventLabel badge when provided', () => {
  const frame = { frame_id: 'f-event', video_id: 'V01', timestamp_ms: 1200 };
  render(<FrameCard frame={frame} eventLabel="E2" />);
  expect(screen.getByText('E2')).toBeTruthy();
  expect(screen.getByText('V01')).toBeTruthy();
});

