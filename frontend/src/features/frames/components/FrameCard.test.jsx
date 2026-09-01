import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import FrameCard from './FrameCard';

test('renders one submit button without changing to a checkmark', () => {
  const frame = {
    frame_id: 'internal-frame-1',
    video_id: 'L21_V001',
    frame_idx: 17794,
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

test('shows the raw alignment score and representative alignment path', () => {
  const frame = {
    frame_id: 'representative-frame',
    video_id: 'L21_V001',
    frame_idx: 17794,
    score: 2.34567,
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

  expect(screen.getByText('Alignment score: 2.346')).toBeTruthy();
  expect(screen.queryByText(/score details/i)).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: /alignment/i }));
  expect(screen.getByText('roll')).toBeTruthy();
});
