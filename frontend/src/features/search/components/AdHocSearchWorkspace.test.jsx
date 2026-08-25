import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFrames } from '../../../api/search';
import AdHocSearchWorkspace from './AdHocSearchWorkspace';

jest.mock('../../../api/search');

beforeEach(() => {
  jest.clearAllMocks();
});

test('clicking Search sends KIS request and renders retrieved frame', async () => {
  const onFrameClick = jest.fn();
  searchFrames.mockResolvedValueOnce({
    results: [{
      rank: 1,
      frame_id: 'frame-1',
      video_id: 'L21_a_b.folder2.L21_V001',
      frame_idx: 42,
      fps: 25,
      timestamp_ms: 1200,
      caption: 'A red boat',
      thumbnail_url: null,
      frame_url: null,
      scores: { final: 0.9 },
    }],
    warnings: [],
    latency_ms: {
      total: 12,
      query_processing: 1,
      query_encoding: 2,
      candidate_retrieval: 5,
      fusion: 1,
      reranking: 0,
      materialization: 3,
    },
  });

  render(
    <AdHocSearchWorkspace
      topK={100}
      setTopK={jest.fn()}
      onFrameClick={onFrameClick}
    />,
  );
  fireEvent.change(screen.getByPlaceholderText(/Start with/), {
    target: { value: '/kis red boat' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query: 'red boat', topK: 100, queryType: 'kis' }),
  ));
  expect(await screen.findByText(/L21_V001, 42/)).toBeTruthy();
  expect(screen.getAllByText('A red boat')).toHaveLength(2);
  fireEvent.click(screen.getByText(/L21_V001, 42/));
  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    video_id: 'L21_a_b.folder2.L21_V001',
    frame_idx: 42,
    fps: 25,
  }));
});
