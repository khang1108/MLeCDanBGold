import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import TrakePathCard from './TrakePathCard';

test('renders one backend path as the historic thumbnail grid without reordering it', () => {
  const path = {
    video_id: 'L21_a_b.folder2.video-8',
    score: 2.875,
    frame_ids: ['late-frame', 'early-frame'],
    frame_idxs: [40, 20],
    timestamps_ms: [4000, 2000],
  };

  const onFrameClick = jest.fn();

  render(
    <TrakePathCard
      events={['person leaves', 'person enters']}
      path={path}
      onSubmit={jest.fn()}
      onFrameClick={onFrameClick}
    />,
  );

  expect(screen.getByRole('heading', { name: /video-8/ })).toBeTruthy();
  expect(document.querySelector('.trake-video-group .frames-grid')).toBeTruthy();
  expect(screen.queryByText('Alignment score: 2.875')).toBeNull();
  expect(screen.getAllByText('person leaves')).toHaveLength(2);
  expect(screen.getAllByText('person enters')).toHaveLength(2);
  expect(screen.getAllByText(/ms/).map((item) => item.textContent)).toEqual(['4000 ms', '2000 ms']);
  expect(screen.getByAltText('Frame late-frame').getAttribute('src'))
    .toMatch(/\/api\/v1\/keyframes\/late-frame$/);
  expect(screen.getByAltText('Frame early-frame').getAttribute('src'))
    .toMatch(/\/api\/v1\/keyframes\/early-frame$/);

  fireEvent.click(screen.getByAltText('Frame late-frame'));
  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    frame_id: 'late-frame',
    frame_idx: 40,
    timestamp_ms: 4000,
    caption: 'person leaves',
  }));
});

test('submits only this path', () => {
  const path = {
    video_id: 'V01',
    score: 3.0,
    frame_ids: ['a1', 'a2'],
    frame_idxs: [10, 20],
    timestamps_ms: [1000, 2000],
  };
  const onSubmit = jest.fn();

  render(<TrakePathCard events={['e1', 'e2']} path={path} onSubmit={onSubmit} />);
  fireEvent.click(screen.getByRole('button', { name: /submit trake path for v01/i }));

  expect(onSubmit).toHaveBeenCalledWith(path);
});
