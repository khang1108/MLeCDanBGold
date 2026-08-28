import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { frameAssetUrl, searchFrames, searchTrake } from '../../../api/search';
import SearchWorkspace, {
  parseRetrievalDescription,
  parseTrakeEvents,
  progressiveSearchIdKey,
} from './SearchWorkspace';

jest.mock('../../../api/search');

beforeEach(() => {
  searchFrames.mockReset();
  searchTrake.mockReset();
  frameAssetUrl.mockImplementation((frameId, asset) => `http://example.test/${frameId}/${asset}`);
  window.sessionStorage.clear();
});

const EVENT_PLACEHOLDER = 'Describe the event, or add E1, E2, ... for TRAKE';
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
  ['a red vehicle passes', 'kis', 'a red vehicle passes'],
])('routes %s through frame search', async (
  description,
  queryType,
  query,
) => {
  searchFrames.mockResolvedValueOnce({
    results: [], warnings: [], latency_ms: { total: 4 },
  });
  render(<SearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit(description);

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query, queryType, topK: 20 }),
  ));
  expect(searchTrake).not.toHaveBeenCalled();
});

test('TRAKE groups clickable event frame cards by video and orders them by frame index', async () => {
  const onFrameClick = jest.fn();
  searchTrake.mockResolvedValueOnce({
    events: ['person enters', 'person leaves'],
    submissions: [
      {
        rank: 2,
        video_id: 'L21_a_b.folder2.video-8',
        frame_ids: ['f40', 'f20'],
        frame_idxs: [40, 20],
        timestamps_ms: [4000, 2000],
        fps: 29.97,
      },
      {
        rank: 1,
        video_id: 'L21_a_b.folder2.video-7',
        frame_ids: ['f30', 'f10'],
        frame_idxs: [30, 10],
        timestamps_ms: [3000, 1000],
        fps: 25,
      },
      {
        rank: 3,
        video_id: 'L21_a_b.folder2.video-7',
        frame_ids: ['f25', 'f5'],
        frame_idxs: [25, 5],
        timestamps_ms: [2500, 500],
        fps: 25,
      },
    ],
    total_results: 3,
    warnings: [],
  });
  render(<SearchWorkspace topK={20} setTopK={jest.fn()} onFrameClick={onFrameClick} />);
  submit('Person enters and leaves.\nE1: person enters\nE2: person leaves');

  await waitFor(() => expect(searchTrake).toHaveBeenCalledWith(
    expect.objectContaining({ events: ['person enters', 'person leaves'], topK: 20 }),
  ));
  expect(searchFrames).not.toHaveBeenCalled();
  expect(await screen.findByRole('heading', { name: /video-7/ })).toBeTruthy();
  expect(screen.getByRole('heading', { name: /video-8/ })).toBeTruthy();
  expect(Array.from(document.querySelectorAll('.trake-video-group h3')).map((heading) => heading.textContent.replace('⬆', '')))
    .toEqual(['video-7', 'video-8']);
  expect(screen.getAllByAltText(/Frame f5|Frame f10|Frame f25|Frame f30/).map((item) => item.alt)).toEqual([
    'Frame f5', 'Frame f10', 'Frame f25', 'Frame f30',
  ]);
  fireEvent.click(screen.getByAltText('Frame f20'));
  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    frame_id: 'f20',
    video_id: 'L21_a_b.folder2.video-8',
    frame_idx: 20,
    timestamp_ms: 2000,
    fps: 29.97,
  }));
});

test('TRAKE requires at least two events without making a request', () => {
  render(<SearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit('A short video. E1: only one event');
  expect(screen.getByRole('alert').textContent).toContain('at least two');
  expect(searchTrake).not.toHaveBeenCalled();
});

test('defaults plain descriptions to KIS', () => {
  expect(parseRetrievalDescription('a red vehicle passes')).toEqual({
    queryType: 'kis',
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
    latency_ms: { total: 4 },
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

test('New Search clears the KIS progressive ID', () => {
  const keys = ['kis'].map(progressiveSearchIdKey);
  keys.forEach((key, index) => window.sessionStorage.setItem(key, `search-${index}`));
  render(<SearchWorkspace topK={20} setTopK={jest.fn()} />);
  fireEvent.click(screen.getByRole('button', { name: 'New Search' }));
  keys.forEach((key) => expect(window.sessionStorage.getItem(key)).toBeNull());
});
