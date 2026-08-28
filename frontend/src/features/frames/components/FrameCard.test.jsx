import React from 'react';
import { render, screen } from '@testing-library/react';
import FrameCard from './FrameCard';

test('renders one submit button without changing to a checkmark', () => {
  const frame = {
    frame_id: 'internal-frame-1',
    video_id: 'L21_V001',
    frame_idx: 17794,
    thumbnail_url: 'https://example.test/frame.jpg',
    caption: 'A sample frame',
  };
  const onSubmit = jest.fn();
  render(<FrameCard frame={frame} onClick={jest.fn()} onSubmit={onSubmit} />);

  const submitButton = screen.getByRole('button', { name: /submit this frame/i });
  expect(screen.getAllByRole('button', { name: /submit this frame/i })).toHaveLength(1);
  expect(submitButton.textContent).toBe('Submit');
  submitButton.click();
  expect(onSubmit).toHaveBeenCalledWith(frame);
  expect(submitButton.textContent).toBe('Submit');
});
