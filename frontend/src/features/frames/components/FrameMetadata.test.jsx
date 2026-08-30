import React from 'react';
import { render, screen } from '@testing-library/react';
import FrameMetadata from './FrameMetadata';

const frame = {
  frame_id: 'f1',
  video_id: 'L21_V001',
  frame_idx: 125,
  timestamp_ms: 5_000,
  fps: 25,
};

test('uses the real playback time for frame index and timestamp metadata', () => {
  render(<FrameMetadata frame={frame} playbackTime={6.24} />);

  expect(screen.getByText('156')).toBeTruthy();
  expect(screen.getByText('6240 ms')).toBeTruthy();
});

test('falls back to canonical metadata before playback time is available', () => {
  render(<FrameMetadata frame={frame} />);

  expect(screen.getByText('125')).toBeTruthy();
  expect(screen.getByText('5000 ms')).toBeTruthy();
});

test('renders the representative metadata without score-stage details', () => {
  render(
    <FrameMetadata
      frame={{
        ...frame,
        score: 1.23456,
        metadata: {
          title: 'Kitchen scene',
          caption: 'A chef coats food',
          ocr: 'FLOUR',
          objects: ['bowl', 'person'],
          asr: 'Coat it with flour',
        },
      }}
    />,
  );

  expect(screen.getByText('Kitchen scene')).toBeTruthy();
  expect(screen.getByText('A chef coats food')).toBeTruthy();
  expect(screen.getByText('FLOUR')).toBeTruthy();
  expect(screen.getByText('bowl, person')).toBeTruthy();
  expect(screen.getByText('Coat it with flour')).toBeTruthy();
  expect(screen.getByText('1.235')).toBeTruthy();
  expect(screen.queryByText(/final relevance/i)).toBeNull();
});
