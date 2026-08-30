import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFrames, searchTrake } from '../../../api/search';
import SearchWorkspace, {
  parseRetrievalDescription,
  parseTrakeEvents,
} from './SearchWorkspace';
import { SubmissionProvider } from '../../submission/contexts/SubmissionContext';

jest.mock('../../../api/search');

beforeEach(() => {
  searchFrames.mockReset();
  searchTrake.mockReset();
});

const EVENT_PLACEHOLDER = 'Describe the event, or add E1, E2, ... for TRAKE';
const SEARCH_LATENCY = {
  query_ms: 1,
  retrieval_ms: 2,
  alignment_ms: 3,
  materialization_ms: 1,
  total_ms: 7,
};
const submit = (eventDescription) => {
  fireEvent.change(screen.getByPlaceholderText(EVENT_PLACEHOLDER), {
    target: { value: eventDescription },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));
};

test('parses sequentially labeled TRAKE events and rejects invalid numbering', () => {
  expect(parseTrakeEvents(
    'A video description. E1: first event E2: second event E3: third event',
  )).toEqual(['first event', 'second event', 'third event']);
  expect(parseTrakeEvents(
    'A video description.\nE1 first event\nE2 second event',
  )).toEqual(['first event', 'second event']);
  expect(parseTrakeEvents('A video description. E1: first event E3: third event')).toEqual([]);
  expect(parseTrakeEvents('A normal KIS description')).toBeNull();
});

test.each([
  ['a red vehicle passes', 'a red vehicle passes'],
])('routes %s through frame search', async (
  description,
  query,
) => {
  searchFrames.mockResolvedValueOnce({
    results: [], warnings: [], latency: SEARCH_LATENCY,
  });
  render(<SearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit(description);

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query, topK: 20 }),
  ));
  expect(searchTrake).not.toHaveBeenCalled();
  expect(await screen.findByText('No frames found matching your query')).toBeTruthy();
});

test('TRAKE renders same-video backend paths independently and submits only the selected path', async () => {
  searchTrake.mockResolvedValueOnce({
    events: ['first event', 'second event'],
    paths: [
      {
        video_id: 'V01',
        score: 3.0,
        frame_ids: ['a1', 'a2'],
        frame_idxs: [10, 20],
        timestamps_ms: [1000, 2000],
        thumbnail_urls: ['/a1', '/a2'],
      },
      {
        video_id: 'V01',
        score: 2.8,
        frame_ids: ['b1', 'b2'],
        frame_idxs: [30, 40],
        timestamps_ms: [3000, 4000],
        thumbnail_urls: ['/b1', '/b2'],
      },
    ],
    total_results: 3,
    warnings: [],
  });
  render(
    <SubmissionProvider>
      <SearchWorkspace topK={20} setTopK={jest.fn()} />
    </SubmissionProvider>,
  );
  submit('Person enters and leaves.\nE1: first event\nE2: second event');

  await waitFor(() => expect(searchTrake).toHaveBeenCalledWith(
    expect.objectContaining({ events: ['first event', 'second event'], topK: 20 }),
  ));
  expect(searchFrames).not.toHaveBeenCalled();
  expect(await screen.findAllByText(/V01/)).toHaveLength(2);
  expect(screen.getByAltText('Frame a1')).toBeTruthy();
  expect(screen.getByAltText('Frame b1')).toBeTruthy();

  fireEvent.click(screen.getAllByRole('button', { name: /submit this path/i })[1]);
  expect(await screen.findByText('V01,30,40')).toBeTruthy();
  expect(screen.queryByText('V01,10,20,30,40')).toBeNull();
});

test('TRAKE requires at least two events without making a request', () => {
  render(<SearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit('A short video. E1: only one event');
  expect(screen.getByRole('alert').textContent).toContain('at least two');
  expect(searchTrake).not.toHaveBeenCalled();
});

test('defaults plain descriptions to KIS', () => {
  expect(parseRetrievalDescription('a red vehicle passes')).toEqual({
    query: 'a red vehicle passes',
  });
});

test('active KIS results preserve backend fps when the user opens a frame', async () => {
  const onFrameClick = jest.fn();
  searchFrames.mockResolvedValueOnce({
    results: [{
      rank: 1,
      frame_id: 'frame-kis',
      video_id: 'L21_a_b.folder2.L21_V001',
      frame_idx: 300,
      fps: 29.97,
      timestamp_ms: 10_010,
      caption: 'A red boat',
      thumbnail_url: 'http://example.test/frame-kis.jpg',
      scores: { final: 0.91 },
    }],
    warnings: [],
    latency: SEARCH_LATENCY,
  });
  render(<SearchWorkspace topK={20} setTopK={jest.fn()} onFrameClick={onFrameClick} />);
  submit('red boat');

  const frameImage = await screen.findByAltText('Frame frame-kis');
  fireEvent.click(frameImage);

  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    video_id: 'L21_a_b.folder2.L21_V001',
    frame_idx: 300,
    fps: 29.97,
  }));
});

test('does not render the retired query-helper control', () => {
  render(<SearchWorkspace topK={20} setTopK={jest.fn()} />);
  expect(screen.queryByRole('button', { name: /suggest query/i })).toBeNull();
});
