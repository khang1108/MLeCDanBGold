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
