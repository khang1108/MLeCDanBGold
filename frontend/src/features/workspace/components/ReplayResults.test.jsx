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

test('replays KIS viewed state without deriving submission state from history', () => {
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
      frameActivity={{ viewed_frame_ids: ['f1'] }}
    />
  );

  expect(screen.getByText((_, element) => element.classList.contains('latency-summary')).textContent)
    .toBe('Found 1 frames in 0.01s');
  expect(screen.getByText('V01')).toBeTruthy();
  expect(screen.getByText('1000 ms')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Alignment' })).toBeTruthy();
  expect(screen.getByAltText('Frame f1').closest('.frame-card').className).toContain('viewed');
  expect(screen.getByAltText('Frame f1').closest('.frame-card').className).not.toContain('submitted');
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
  );
});

test('shows the unsupported-history message for a saved legacy paths snapshot', () => {
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
    />
  );

  expect(screen.getByText('This history snapshot cannot be replayed.')).toBeTruthy();
  expect(screen.queryByRole('button', { name: /event/i })).toBeNull();
});
