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

test('uses the real playback time for timestamp metadata', () => {
  render(<FrameMetadata frame={frame} playbackTime={6.24} />);

  expect(screen.getByText('L21_V001')).toBeTruthy();
  expect(screen.getByText('6240 ms')).toBeTruthy();
});

test('falls back to canonical metadata before playback time is available', () => {
  render(<FrameMetadata frame={frame} />);

  expect(screen.getByText('L21_V001')).toBeTruthy();
  expect(screen.getByText('5000 ms')).toBeTruthy();
});

test('renders only Video ID and Timestamp without ASR, objects, captions, or score', () => {
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

  expect(screen.getByText('L21_V001')).toBeTruthy();
  expect(screen.getByText('5000 ms')).toBeTruthy();
  expect(screen.queryByText('Kitchen scene')).toBeNull();
  expect(screen.queryByText('A chef coats food')).toBeNull();
  expect(screen.queryByText('FLOUR')).toBeNull();
  expect(screen.queryByText('bowl, person')).toBeNull();
  expect(screen.queryByText('Coat it with flour')).toBeNull();
  expect(screen.queryByText('1.235')).toBeNull();
});
