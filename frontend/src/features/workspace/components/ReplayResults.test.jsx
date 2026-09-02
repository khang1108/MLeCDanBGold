import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ReplayResults from './ReplayResults';

const latency = {
  query_ms: 1,
  retrieval_ms: 2,
  alignment_ms: 3,
  materialization_ms: 1,
  total_ms: 7,
};

test('renders KIS with the legacy FramesBox and FrameCard UI', () => {
  render(
    <ReplayResults
      resultSnapshot={{
        events: ['person enters'],
        latency,
        warnings: [],
        results: [{
          frame_id: 'f1',
          video_id: 'V01',
          frame_idx: 10,
          timestamp_ms: 1000,
          score: 0.9,
          frame_ids: ['f1'],
          timestamps_ms: [1000],
          caption: 'A person enters',
          metadata: {
            title: 'video title',
            caption: 'A person enters',
            ocr: 'visible text',
            objects: ['person'],
            asr: 'spoken words',
          },
        }],
      }}
      frameActivity={{ viewed_frame_ids: ['f1'], submitted_frame_ids: ['f1'] }}
    />
  );

  expect(screen.getByText((_, element) => element.classList.contains('latency-summary')).textContent)
    .toBe('Found 1 frames in 7ms');
  expect(screen.getByText('V01, 10')).toBeTruthy();
  expect(screen.getAllByText('A person enters')).toHaveLength(2);
  expect(screen.getByText('Alignment score: 0.900')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Alignment' })).toBeTruthy();
  expect(screen.getByAltText('Frame f1').closest('.frame-card').className).toContain('submitted');
  expect(screen.queryByText('Replay')).toBeNull();
});

test('opens ImageModal from stored search metadata without a frame-detail request', () => {
  const onFrameClick = jest.fn();
  render(
    <ReplayResults
      resultSnapshot={{
        events: [],
        latency,
        warnings: [],
        results: [{
          frame_id: 'f1',
          video_id: 'V01',
          frame_idx: 10,
          timestamp_ms: 1000,
          score: 0.9,
          frame_ids: ['f1'],
          timestamps_ms: [1000],
          caption: 'A person enters',
          metadata: {
            title: 'video title',
            caption: 'A person enters',
            ocr: 'visible text',
            objects: ['person'],
            asr: 'spoken words',
          },
        }],
      }}
      onFrameClick={onFrameClick}
    />
  );

  fireEvent.click(screen.getByAltText('Frame f1'));
  expect(onFrameClick).toHaveBeenCalledWith(
    expect.objectContaining({
      frame_id: 'f1',
      video_id: 'V01',
      metadata: expect.objectContaining({ ocr: 'visible text' }),
    }),
    'kis',
  );
});

test('renders TRAKE with the legacy ordered event rows and submits the selected path', () => {
  const onPathSubmit = jest.fn();
  const onFrameClick = jest.fn();
  render(
    <ReplayResults
      resultSnapshot={{
        events: ['person enters', 'person exits'],
        latency,
        warnings: [],
        paths: [{
          video_id: 'V01',
          score: 1.2,
          frame_ids: ['f1', 'f2'],
          frame_idxs: [10, 20],
          timestamps_ms: [1000, 2000],
        }],
      }}
      frameActivity={{ viewed_frame_ids: ['f2'], submitted_frame_ids: ['f1'] }}
      onPathSubmit={onPathSubmit}
      onFrameClick={onFrameClick}
    />
  );

  expect(screen.getByRole('button', { name: 'View event E1: person enters' })).toBeTruthy();
  expect(screen.getByText('person exits')).toBeTruthy();
  expect(screen.getByText('1000 ms')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Submit this path' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'View event E1: person enters' }).className).toContain('submitted');
  expect(screen.getByRole('button', { name: 'View event E2: person exits' }).className).toContain('viewed');

  fireEvent.click(screen.getByRole('button', { name: 'Submit this path' }));
  expect(onPathSubmit).toHaveBeenCalledWith(expect.objectContaining({
    video_id: 'V01',
    frame_ids: ['f1', 'f2'],
    frame_idxs: [10, 20],
  }));

  fireEvent.click(screen.getByRole('button', { name: 'View event E1: person enters' }));
  expect(onFrameClick).toHaveBeenCalledWith(
    expect.objectContaining({ frame_id: 'f1', frame_idx: 10 }),
    'none',
  );
});
