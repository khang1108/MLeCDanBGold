import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFrames } from '../../../api/search';
import SearchWorkspace, { parseRetrievalDescription } from './SearchWorkspace';
import AnswerWorkspaceProvider from '../../answer-workspace/contexts/AnswerWorkspaceContext';
import {
  createQueryHistory,
  markFrameViewed,
} from '../../../api/history';

jest.mock('../../../api/search');
jest.mock('../../../api/history', () => ({
  createQueryHistory: jest.fn(),
  markFrameViewed: jest.fn(),
}));
const renderSearch = (props) => render(
  <AnswerWorkspaceProvider connectedUserId="">
    <SearchWorkspace {...props} />
  </AnswerWorkspaceProvider>,
);

beforeEach(() => {
  searchFrames.mockReset();
  createQueryHistory.mockResolvedValue({});
  markFrameViewed.mockResolvedValue({});
});

const SEARCH_LATENCY = {
  query_ms: 1,
  retrieval_ms: 2,
  alignment_ms: 3,
  materialization_ms: 1,
  total_ms: 7,
};
const submit = (eventDescription) => {
  fireEvent.change(document.getElementById('event-query'), {
    target: { value: eventDescription },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));
};

test.each([
  ['a red vehicle passes', 'a red vehicle passes'],
  ['E1: a person enters the room', 'E1: a person enters the room'],
])('routes %s through frame search', async (
  description,
  query,
) => {
  searchFrames.mockResolvedValueOnce({
    results: [], warnings: [], latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn() });
  submit(description);

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query, topK: 20 }),
  ));
  expect(await screen.findByText('No frames found matching your query')).toBeTruthy();
});

test('Enter submits E1-prefixed text while Shift+Enter stays in the textarea', async () => {
  searchFrames.mockResolvedValueOnce({
    results: [], warnings: [], latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn() });
  const textarea = document.getElementById('event-query');
  fireEvent.change(textarea, { target: { value: 'E1: a person enters the room' } });

  expect(fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter', shiftKey: true })).toBe(true);
  expect(searchFrames).not.toHaveBeenCalled();
  expect(fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter', shiftKey: false })).toBe(false);

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query: 'E1: a person enters the room', topK: 20 }),
  ));
});

test('sends the selected Dense and BM25 modes with KIS search', async () => {
  searchFrames.mockResolvedValueOnce({
    results: [], warnings: [], latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn() });

  fireEvent.click(screen.getByRole('switch', { name: /use dense retrieval/i }));
  submit('a lexical-only query');

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({
      query: 'a lexical-only query',
      topK: 20,
      useDense: false,
      useBm25: true,
    }),
  ));
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
      frame_ids: ['frame-kis'],
      fps: 29.97,
      timestamp_ms: 10_010,
      caption: 'A red boat',
      scores: { final: 0.91 },
    }],
    warnings: [],
    latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn(), onFrameClick });
  submit('red boat');

  const frameImage = await screen.findByAltText('Frame frame-kis');
  fireEvent.click(frameImage);

  expect(onFrameClick).toHaveBeenCalledWith({
    frame: expect.objectContaining({
      video_id: 'L21_a_b.folder2.L21_V001',
      frame_idx: 300,
      fps: 29.97,
    }),
  });
});

test('passes the immutable live KIS scoring snapshot when opening a result', async () => {
  const onFrameClick = jest.fn();
  const response = {
    results: [{
      frame_id: 'frame-explore',
      video_id: 'V01',
      frame_idx: 125,
      timestamp_ms: 10_010,
      fps: 29.97,
      caption: 'A red boat',
      scores: { final: 0.91 },
    }],
    events: ['red boat'],
    dense_events: ['dense red boat'],
    bm25_caption_events: ['caption red boat'],
    use_dense: true,
    use_bm25: true,
    warnings: [],
    latency: SEARCH_LATENCY,
  };
  searchFrames.mockResolvedValueOnce(response);
  renderSearch({ topK: 20, setTopK: jest.fn(), onFrameClick });
  submit('red boat');

  fireEvent.change(document.getElementById('event-query'), {
    target: { value: 'a different draft query' },
  });
  fireEvent.click(await screen.findByAltText('Frame frame-explore'));

  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    frame: expect.objectContaining({ frame_id: 'frame-explore', video_id: 'V01' }),
    explorationSnapshot: expect.objectContaining({
      query: 'red boat',
      events: ['red boat'],
      dense_events: ['dense red boat'],
      bm25_caption_events: ['caption red boat'],
      use_dense: true,
      use_bm25: true,
    }),
  }));
});

test('keeps a saved legacy paths snapshot on the unsupported-history message', async () => {
  renderSearch({
    topK: 20,
    setTopK: jest.fn(),
    userId: 'team-a',
    replayRequest: {
      token: 3,
      item: {
        query_id: 'legacy-query',
        query_text: 'saved query',
        result_snapshot: { paths: [] },
        frame_activity: {},
      },
    },
  });

  expect(await screen.findByText('This history snapshot cannot be replayed.')).toBeTruthy();
});

test('does not render the retired query-helper control', () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });
  expect(screen.queryByRole('button', { name: /suggest query/i })).toBeNull();
});

test('renders the shared answer workspace in the KIS sidebar', () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });

  expect(screen.getByRole('region', { name: 'Answer workspace' })).toBeTruthy();
});

test('adds a result frame using its exact timestamp when frame_idx differs', async () => {
  const onAddCandidate = jest.fn();
  searchFrames.mockResolvedValueOnce({
    results: [{
      frame_id: 'search-time-frame',
      video_id: 'V01',
      frame_idx: 3,
      timestamp_ms: 12_345,
      frame_ids: ['search-time-frame'],
      timestamps_ms: [12_345],
      scores: { final: 0.9 },
    }],
    warnings: [],
    latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn(), onAddCandidate });
  submit('boat');

  fireEvent.click(await screen.findByRole('button', { name: 'Add frame to answer workspace' }));
  expect(onAddCandidate).toHaveBeenCalledWith({ kind: 'FRAME', videoId: 'V01', timestampMs: 12_345 });
});

test('keeps local retrieval available while no VBS participant is connected', async () => {
  const onFocusUserId = jest.fn();
  searchFrames.mockResolvedValueOnce({
    results: [], warnings: [], latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: '  ', onFocusUserId });
  submit('a red vehicle passes');

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query: 'a red vehicle passes', userId: '' }),
  ));
  expect(onFocusUserId).not.toHaveBeenCalled();
  expect(await screen.findByText('No frames found matching your query')).toBeTruthy();
  expect(createQueryHistory).not.toHaveBeenCalled();
});

test('keeps disconnected retrieval in the draft user history without sending the DRES header', async () => {
  const { searchFrames: realSearchFrames } = jest.requireActual('../../../api/search');
  const frame = {
    frame_id: 'disconnected-frame',
    video_id: 'V01',
    frame_idx: 4,
    timestamp_ms: 160,
    frame_ids: ['disconnected-frame'],
    timestamps_ms: [160],
    scores: { final: 0.8 },
    caption: 'A boat crosses the scene',
  };
  searchFrames.mockImplementation(realSearchFrames);
  const fetchSpy = jest.spyOn(global, 'fetch').mockResolvedValue({
    ok: true,
    status: 200,
    json: jest.fn().mockResolvedValue({
      events: [],
      results: [frame],
      latency: SEARCH_LATENCY,
    }),
  });
  renderSearch({
    topK: 20,
    setTopK: jest.fn(),
    userId: '',
    historyUserId: 'team-a',
  });
  submit('a boat crosses the scene');

  const resultImage = await screen.findByAltText('Frame disconnected-frame');
  await waitFor(() => expect(createQueryHistory).toHaveBeenCalledWith(expect.objectContaining({
    userId: 'team-a',
    queryText: 'a boat crosses the scene',
  })));
  expect(searchFrames).toHaveBeenCalledWith(expect.objectContaining({ userId: '' }));
  const searchCall = fetchSpy.mock.calls.find(([url]) => String(url).includes('/api/v1/search'));
  expect(searchCall[1].headers['X-VBS-User-ID']).toBeUndefined();

  fireEvent.click(resultImage);
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledWith({
    queryId: createQueryHistory.mock.calls[0][0].queryId,
    frameId: 'disconnected-frame',
  }));
  fetchSpy.mockRestore();
});

test('persists a successful KIS search as a full replay snapshot', async () => {
  searchFrames.mockResolvedValueOnce({
    results: [{
      frame_id: 'frame-kis',
      video_id: 'V01',
      frame_idx: 125,
      timestamp_ms: 10_010,
      fps: 29.97,
      folder_id: 'L21',
      frame_ids: ['frame-kis'],
      timestamps_ms: [10_010],
      scores: { final: 0.91 },
      caption: 'A red boat',
      metadata: {
        title: 'boat video',
        caption: 'A red boat',
        ocr: 'MARINA',
        objects: ['boat', 'person'],
        asr: 'A boat is moving',
      },
    }],
    warnings: [],
    latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('red boat');

  await waitFor(() => expect(createQueryHistory).toHaveBeenCalledWith(expect.objectContaining({
    userId: 'team-a',
    queryText: 'red boat',
    resultSnapshot: {
      events: [],
      latency: SEARCH_LATENCY,
      warnings: [],
      results: [{
        frame_id: 'frame-kis',
        video_id: 'V01',
        frame_idx: 125,
        timestamp_ms: 10_010,
        fps: 29.97,
        folder_id: 'L21',
        scores: { final: 0.91 },
        score: 0.91,
        frame_ids: ['frame-kis'],
        timestamps_ms: [10_010],
        caption: 'A red boat',
        metadata: {
          title: 'boat video',
          caption: 'A red boat',
          ocr: 'MARINA',
          objects: ['boat', 'person'],
          asr: 'A boat is moving',
        },
      }],
    },
    signal: expect.any(AbortSignal),
  })));
  expect(createQueryHistory.mock.calls[0][0].queryId).toMatch(/^query-/);
  expect(createQueryHistory.mock.calls[0][0].resultSnapshot.results[0].metadata).toEqual({
    title: 'boat video',
    caption: 'A red boat',
    ocr: 'MARINA',
    objects: ['boat', 'person'],
    asr: 'A boat is moving',
  });
});

test('keeps live results visible but creates no active history session when history persistence fails', async () => {
  createQueryHistory.mockRejectedValueOnce(new Error('history unavailable'));
  searchFrames.mockResolvedValueOnce({
    results: [{
      frame_id: 'frame-kis',
      video_id: 'V01',
      frame_idx: 1,
      timestamp_ms: 1000,
      frame_ids: ['frame-kis'],
      timestamps_ms: [1000],
      scores: { final: 0.5 },
    }],
    warnings: [],
    latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('red boat');

  expect(await screen.findByAltText('Frame frame-kis')).toBeTruthy();
  expect(await screen.findByText(/history was not saved/i)).toBeTruthy();
  fireEvent.click(screen.getByAltText('Frame frame-kis'));
  expect(markFrameViewed).not.toHaveBeenCalled();
});

test('keeps the viewed color while allowing a failed activity patch to retry', async () => {
  markFrameViewed.mockRejectedValueOnce(new Error('activity unavailable'));
  searchFrames.mockResolvedValueOnce({
    results: [{
      frame_id: 'frame-kis',
      video_id: 'V01',
      frame_idx: 1,
      timestamp_ms: 1000,
      frame_ids: ['frame-kis'],
      timestamps_ms: [1000],
      scores: { final: 0.5 },
    }],
    warnings: [],
    latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('red boat');

  const frameImage = await screen.findByAltText('Frame frame-kis');
  fireEvent.click(frameImage);
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledTimes(1));
  expect(await screen.findByText(/history view state was not recorded/i)).toBeTruthy();
  fireEvent.click(frameImage);
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledTimes(2));
});
