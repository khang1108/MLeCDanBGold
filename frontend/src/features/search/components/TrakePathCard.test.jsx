import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import TrakePathCard from './TrakePathCard';

test('renders one backend path in event order with its raw score and thumbnails', () => {
  const path = {
    video_id: 'L21_a_b.folder2.video-8',
    score: 2.875,
    frame_ids: ['late-frame', 'early-frame'],
    frame_idxs: [40, 20],
    timestamps_ms: [4000, 2000],
    thumbnail_urls: ['/late-frame', '/early-frame'],
  };

  render(
    <TrakePathCard
      events={['person leaves', 'person enters']}
      path={path}
      onSubmit={jest.fn()}
    />,
  );

  expect(screen.getByRole('heading', { name: 'video-8' })).toBeTruthy();
  expect(screen.getByText('Alignment score: 2.875')).toBeTruthy();
  expect(screen.getAllByText(/E[12]/).map((item) => item.textContent)).toEqual(['E1', 'E2']);
  expect(screen.getByText('person leaves')).toBeTruthy();
  expect(screen.getByText('person enters')).toBeTruthy();
  expect(screen.getAllByText(/ms/).map((item) => item.textContent)).toEqual(['4000 ms', '2000 ms']);
  expect(screen.getByAltText('Frame late-frame').getAttribute('src')).toBe('/late-frame');
  expect(screen.getByAltText('Frame early-frame').getAttribute('src')).toBe('/early-frame');
});

test('submits only this path', () => {
  const path = {
    video_id: 'V01',
    score: 3.0,
    frame_ids: ['a1', 'a2'],
    frame_idxs: [10, 20],
    timestamps_ms: [1000, 2000],
    thumbnail_urls: ['/a1', '/a2'],
  };
  const onSubmit = jest.fn();

  render(<TrakePathCard events={['e1', 'e2']} path={path} onSubmit={onSubmit} />);
  fireEvent.click(screen.getByRole('button', { name: /submit this path/i }));

  expect(onSubmit).toHaveBeenCalledWith(path);
});
