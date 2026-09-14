import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import WorkspaceInlinePlayer from './WorkspaceInlinePlayer';

const sampleFrame = {
  frame_id: 'L21_V001_00000300',
  video_id: 'L21_V001',
  frame_idx: 300,
  timestamp_ms: 12040,
  fps: 25,
  metadata: {
    caption: 'A traffic scene with cars and bicycles',
    ocr: 'SHOP 24/7',
    asr: 'Speaking about traffic',
    objects: ['car', 'bicycle'],
  },
};

describe('WorkspaceInlinePlayer', () => {
  test('renders video stream with title and timestamp without captions or metadata cards', () => {
    const onClose = jest.fn();
    render(
      <WorkspaceInlinePlayer
        frame={sampleFrame}
        initialTimestampMs={12000}
        onClose={onClose}
      />,
    );

    // Header title with video id
    expect(screen.getByText('L21_V001')).toBeTruthy();
    expect(screen.getByText('12040 ms')).toBeTruthy();

    // Close button
    const closeBtn = screen.getByRole('button', { name: 'Close video player' });
    expect(closeBtn).toBeTruthy();
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalledTimes(1);

    // Video element
    const video = screen.getByLabelText('Video for L21_V001');
    expect(video).toBeTruthy();
    expect(video.getAttribute('src')).toContain('/videos/L21_V001/stream');

    // Verify captions/metadata are NOT rendered
    expect(screen.queryByText('A traffic scene with cars and bicycles')).toBeNull();
    expect(screen.queryByText('SHOP 24/7')).toBeNull();
    expect(screen.queryByText('Speaking about traffic')).toBeNull();
    expect(screen.queryByText(/bicycle/i)).toBeNull();
  });

  test('displays error message when video stream url cannot be resolved', () => {
    render(
      <WorkspaceInlinePlayer
        frame={{ video_id: '', timestamp_ms: null }}
      />,
    );

    expect(screen.getByText(/video playback is unavailable/i)).toBeTruthy();
  });
});
