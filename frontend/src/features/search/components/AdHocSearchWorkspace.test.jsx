import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFrames } from '../../../api/search';
import AdHocSearchWorkspace from './AdHocSearchWorkspace';

jest.mock('../../../api/search');

test('clicking Search sends KIS request and renders retrieved frame', async () => {
  searchFrames.mockResolvedValueOnce({
    results: [{
      rank: 1,
      frame_id: 'frame-1',
      video_id: 'video-1',
      frame_idx: 42,
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
      onFrameClick={jest.fn()}
    />,
  );
  fireEvent.change(screen.getByPlaceholderText(/Search frames/), {
    target: { value: 'red boat' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query: 'red boat', topK: 100 }),
  ));
  expect(await screen.findByText(/video-1 · frame 42/)).toBeTruthy();
  expect(screen.getAllByText('A red boat')).toHaveLength(2);
});
