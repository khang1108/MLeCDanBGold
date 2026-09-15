import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFramesByImage } from '../../../api/search';
import { searchKis } from '../../../api/kis';
import SearchWorkspace, { parseRetrievalDescription } from './SearchWorkspace';
import { filterFrames } from '../../../api/filter';
import {
  createQueryHistory,
  markFrameViewed,
} from '../../../api/history';

jest.mock('../../../api/search', () => ({
  ...jest.requireActual('../../../api/search'),
  searchFramesByImage: jest.fn(),
}));
jest.mock('../../../api/kis', () => ({
  searchKis: jest.fn(),
}));
jest.mock('../../../api/filter', () => ({
  filterFrames: jest.fn(),
}));
jest.mock('../../../api/history', () => ({
  createQueryHistory: jest.fn(),
  markFrameViewed: jest.fn(),
}));
const renderSearch = (props) => render(<SearchWorkspace {...props} />);

beforeEach(() => {
  searchKis.mockReset();
  searchFramesByImage.mockReset();
  filterFrames.mockReset();
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

const mockKisResponse = ({
  results = [],
  inputs = ['test'],
  events = [{ id: 'E1', text: 'test' }],
  queryText = 'test',
  revision = 1,
  warnings = [],
  latency = SEARCH_LATENCY,
} = {}) => ({
  intent: {
    revision,
    inputs,
    language: 'en',
    query_text: queryText,
    entities: [],
    events,
    temporal_edges: [],
  },
  results,
  warnings,
  latency,
});

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
  searchKis.mockResolvedValueOnce(mockKisResponse({ inputs: [query], queryText: query }));
  renderSearch({ topK: 20, setTopK: jest.fn() });
  submit(description);

  await waitFor(() => expect(searchKis).toHaveBeenCalledWith(
    expect.objectContaining({ inputs: [{ text: query }], topK: 20 }),
  ));
  expect(await screen.findByText('No frames found matching your query')).toBeTruthy();
});

test('Enter submits E1-prefixed text while Shift+Enter stays in the textarea', async () => {
  searchKis.mockResolvedValueOnce(mockKisResponse({
    inputs: ['E1: a person enters the room'],
    queryText: 'E1: a person enters the room',
  }));
  renderSearch({ topK: 20, setTopK: jest.fn() });
  const textarea = document.getElementById('event-query');
  fireEvent.change(textarea, { target: { value: 'E1: a person enters the room' } });

  expect(fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter', shiftKey: true })).toBe(true);
  expect(searchKis).not.toHaveBeenCalled();
  expect(fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter', shiftKey: false })).toBe(false);

  await waitFor(() => expect(searchKis).toHaveBeenCalledWith(
    expect.objectContaining({ inputs: [{ text: 'E1: a person enters the room' }], topK: 20 }),
  ));
});

test('sends the selected Dense and BM25 modes with KIS search', async () => {
  searchKis.mockResolvedValueOnce(mockKisResponse({
    inputs: ['a lexical-only query'],
    queryText: 'a lexical-only query',
  }));
  renderSearch({ topK: 20, setTopK: jest.fn() });

  fireEvent.click(screen.getByRole('switch', { name: /use dense retrieval/i }));
  submit('a lexical-only query');

  await waitFor(() => expect(searchKis).toHaveBeenCalledWith(
    expect.objectContaining({
      inputs: [{ text: 'a lexical-only query' }],
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

test('renders one KIS clue input owned by the unified panel', () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });

  expect(screen.getAllByPlaceholderText('Search or add another clue…')).toHaveLength(1);
  expect(screen.queryByLabelText('KIS Chat Assistant')).toBeNull();
});

test('active KIS results preserve backend fps when the user opens a frame', async () => {
  const onFrameClick = jest.fn();
  searchKis.mockResolvedValueOnce(mockKisResponse({
    inputs: ['red boat'],
    queryText: 'red boat',
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
  }));
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
    ...mockKisResponse({
      inputs: ['red boat'],
      queryText: 'red boat',
      events: [{ id: 'E1', text: 'red boat' }],
      results: [{
        frame_id: 'frame-explore',
        video_id: 'V01',
        frame_idx: 125,
        timestamp_ms: 10_010,
        fps: 29.97,
        caption: 'A red boat',
        scores: { final: 0.91 },
      }],
    }),
    exploration_seed: {
      semantic_revision: 1,
      events: [{ event_id: 'E1', canonical_text: 'red boat', dense_text: 'dense red boat', bm25_text: 'caption red boat' }],
      use_dense: true,
      use_bm25: true,
    },
    use_dense: true,
    use_bm25: true,
  };
  searchKis.mockResolvedValueOnce(response);
  renderSearch({ topK: 20, setTopK: jest.fn(), onFrameClick });
  submit('red boat');

  fireEvent.change(document.getElementById('event-query'), {
    target: { value: 'a different draft query' },
  });
  fireEvent.click(await screen.findByAltText('Frame frame-explore'));

  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    frame: expect.objectContaining({ frame_id: 'frame-explore', video_id: 'V01' }),
    explorationSnapshot: response.exploration_seed,
  }));
});

const frameResult = (id) => ({
  frame_id: id,
  video_id: 'V01',
  frame_idx: id === 'frame-1' ? 100 : 200,
  timestamp_ms: id === 'frame-1' ? 4_000 : 8_000,
  fps: 25,
  caption: id,
  scores: { final: 0.9 },
});

test('keeps committed results visible while the next revision is pending', async () => {
  let resolveSecond;
  searchKis
    .mockResolvedValueOnce(mockKisResponse({ queryText: 'first', revision: 1, results: [frameResult('frame-1')] }))
    .mockImplementationOnce(() => new Promise((resolve) => { resolveSecond = resolve; }));

  renderSearch({ topK: 20, setTopK: jest.fn() });
  submit('first');
  expect(await screen.findByAltText('Frame frame-1')).toBeTruthy();

  submit('second');
  expect(screen.getByAltText('Frame frame-1')).toBeTruthy();

  resolveSecond(mockKisResponse({ queryText: 'second', revision: 2, results: [frameResult('frame-2')] }));
  expect(await screen.findByAltText('Frame frame-2')).toBeTruthy();
});

test('keeps previous committed results and retains draft when next revision fails', async () => {
  searchKis
    .mockResolvedValueOnce(mockKisResponse({ queryText: 'first', revision: 1, results: [frameResult('frame-1')] }))
    .mockRejectedValueOnce(new Error('Network error (500)'));

  renderSearch({ topK: 20, setTopK: jest.fn() });
  submit('first');
  expect(await screen.findByAltText('Frame frame-1')).toBeTruthy();

  submit('second clue');
  expect(await screen.findByText('Network error (500)')).toBeTruthy();
  expect(screen.getByAltText('Frame frame-1')).toBeTruthy();
  const input = document.getElementById('event-query');
  expect(input.value).toBe('second clue');
});

test('opens a KIS result exactly once and records one viewed-frame write', async () => {
  const onFrameClick = jest.fn();
  const response = {
    ...mockKisResponse({
      queryText: 'red boat',
      revision: 1,
      results: [{
        frame_id: 'frame-explore',
        video_id: 'V01',
        frame_idx: 125,
        timestamp_ms: 10_010,
        fps: 29.97,
        caption: 'A red boat',
        scores: { final: 0.91 },
      }],
    }),
    exploration_seed: {
      semantic_revision: 1,
      events: [{ event_id: 'E1', canonical_text: 'red boat', dense_text: 'dense red boat', bm25_text: 'caption red boat' }],
      use_dense: true,
      use_bm25: true,
    },
  };
  searchKis.mockResolvedValueOnce(response);
  renderSearch({ topK: 20, setTopK: jest.fn(), onFrameClick, userId: 'team-a' });
  submit('red boat');
  const frame = await screen.findByAltText('Frame frame-explore');
  fireEvent.click(frame);

  expect(onFrameClick).toHaveBeenCalledTimes(1);
  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    explorationSnapshot: expect.any(Object),
  }));
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledTimes(1));
  expect(markFrameViewed).toHaveBeenCalledWith(expect.objectContaining({
    frameId: 'frame-explore',
  }));
});

test('a frame clicked before createQueryHistory resolves does not call markFrameViewed until creation resolves', async () => {
  let resolveHistory;
  createQueryHistory.mockImplementationOnce(() => new Promise((resolve) => { resolveHistory = resolve; }));
  searchKis.mockResolvedValueOnce(mockKisResponse({
    queryText: 'first',
    revision: 1,
    results: [frameResult('frame-1')],
  }));

  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('first');
  const frame = await screen.findByAltText('Frame frame-1');
  fireEvent.click(frame);

  expect(markFrameViewed).not.toHaveBeenCalled();

  resolveHistory({});
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledTimes(1));
  expect(markFrameViewed).toHaveBeenCalledWith(expect.objectContaining({
    frameId: 'frame-1',
  }));
});

test('skips a queued viewed-frame write when query history creation fails', async () => {
  let rejectHistory;
  createQueryHistory.mockImplementationOnce(() => new Promise((resolve, reject) => {
    rejectHistory = reject;
  }));
  searchKis.mockResolvedValueOnce(mockKisResponse({
    queryText: 'first',
    revision: 1,
    results: [frameResult('frame-1')],
  }));

  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('first');
  const frame = await screen.findByAltText('Frame frame-1');
  fireEvent.click(frame);

  expect(markFrameViewed).not.toHaveBeenCalled();
  await act(async () => {
    rejectHistory(new Error('history unavailable'));
    await Promise.resolve();
    await Promise.resolve();
  });

  expect(markFrameViewed).not.toHaveBeenCalled();
  expect(await screen.findByText(/history was not saved: history unavailable/i)).toBeTruthy();
  expect(screen.getByAltText('Frame frame-1')).toBeTruthy();
});

test('late history completion from search A cannot overwrite active search B', async () => {
  let resolveHistoryA;
  createQueryHistory
    .mockImplementationOnce(() => new Promise((resolve) => { resolveHistoryA = resolve; }))
    .mockResolvedValueOnce({});

  searchKis
    .mockResolvedValueOnce(mockKisResponse({ queryText: 'first', revision: 1, results: [frameResult('frame-1')] }))
    .mockResolvedValueOnce(mockKisResponse({ queryText: 'second', revision: 2, results: [frameResult('frame-2')] }));

  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('first');
  await screen.findByAltText('Frame frame-1');

  submit('second');
  await screen.findByAltText('Frame frame-2');

  const queryIdB = createQueryHistory.mock.calls[1][0].queryId;

  resolveHistoryA({});

  fireEvent.click(screen.getByAltText('Frame frame-2'));
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledWith(expect.objectContaining({
    queryId: queryIdB,
    frameId: 'frame-2',
  })));
});

test('late history failure from search A cannot warn about or clear active search B', async () => {
  let rejectHistoryA;
  createQueryHistory
    .mockImplementationOnce(() => new Promise((resolve, reject) => { rejectHistoryA = reject; }))
    .mockResolvedValueOnce({});

  searchKis
    .mockResolvedValueOnce(mockKisResponse({ queryText: 'first', revision: 1, results: [frameResult('frame-1')] }))
    .mockResolvedValueOnce(mockKisResponse({ queryText: 'second', revision: 2, results: [frameResult('frame-2')] }));

  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('first');
  await screen.findByAltText('Frame frame-1');
  submit('second');
  await screen.findByAltText('Frame frame-2');

  await act(async () => {
    rejectHistoryA(new Error('search A history unavailable'));
    await Promise.resolve();
    await Promise.resolve();
  });

  expect(screen.getByAltText('Frame frame-2')).toBeTruthy();
  expect(screen.queryByText(/search A history unavailable/i)).toBeNull();
  fireEvent.click(screen.getByAltText('Frame frame-2'));
  const queryIdB = createQueryHistory.mock.calls[1][0].queryId;
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledWith(expect.objectContaining({
    queryId: queryIdB,
    frameId: 'frame-2',
  })));
});

test('typing a new draft does not invoke onQueryChange; successful search calls onQueryChange with committed queryText', async () => {
  const onQueryChange = jest.fn();
  searchKis.mockResolvedValueOnce(mockKisResponse({
    queryText: 'committed query text',
    revision: 1,
    results: [frameResult('frame-1')],
  }));

  renderSearch({ topK: 20, setTopK: jest.fn(), onQueryChange });

  const input = document.getElementById('event-query');
  fireEvent.change(input, { target: { value: 'draft typing...' } });
  expect(onQueryChange).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Search' }));
  await waitFor(() => expect(onQueryChange).toHaveBeenCalledWith('committed query text'));
});

test('replay is read-only and does not copy historical query into draft input', async () => {
  renderSearch({
    topK: 20,
    setTopK: jest.fn(),
    userId: 'team-a',
    replayRequest: {
      token: 1,
      item: {
        query_id: 'q-hist',
        query_text: 'historical clue text',
        result_snapshot: { results: [frameResult('frame-1')] },
        frame_activity: {},
      },
    },
  });

  const input = document.getElementById('event-query');
  expect(input.value).toBe('');
  expect(input.disabled).toBe(true);
  expect(await screen.findByAltText('Frame frame-1')).toBeTruthy();

  fireEvent.click(screen.getByRole('button', { name: 'New Search' }));
  expect(screen.queryByAltText('Frame frame-1')).toBeNull();
  expect(input.value).toBe('');
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

test('does not render a shared answer workspace in the search sidebar', () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });

  expect(screen.queryByRole('region', { name: /answer workspace/i })).toBeNull();
});

test('opens one direct answer from a result frame using its exact timestamp', async () => {
  const onOpenSubmission = jest.fn();
  searchKis.mockResolvedValueOnce(mockKisResponse({
    inputs: ['boat'],
    queryText: 'boat',
    results: [{
      frame_id: 'search-time-frame',
      video_id: 'V01',
      frame_idx: 3,
      timestamp_ms: 12_345,
      frame_ids: ['search-time-frame'],
      timestamps_ms: [12_345],
      scores: { final: 0.9 },
    }],
  }));
  renderSearch({ topK: 20, setTopK: jest.fn(), onOpenSubmission });
  submit('boat');

  fireEvent.click(await screen.findByRole('button', { name: 'Submit this frame to DRES' }));
  expect(onOpenSubmission).toHaveBeenCalledWith({ videoId: 'V01', startMs: 12_345, endMs: 12_345 });
});

test('keeps local retrieval available while no VBS participant is connected', async () => {
  const onFocusUserId = jest.fn();
  searchKis.mockResolvedValueOnce(mockKisResponse({
    inputs: ['a red vehicle passes'],
    queryText: 'a red vehicle passes',
  }));
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: '  ', onFocusUserId });
  submit('a red vehicle passes');

  await waitFor(() => expect(searchKis).toHaveBeenCalledWith(
    expect.objectContaining({ inputs: [{ text: 'a red vehicle passes' }], userId: '' }),
  ));
  expect(onFocusUserId).not.toHaveBeenCalled();
  expect(await screen.findByText('No frames found matching your query')).toBeTruthy();
  expect(createQueryHistory).not.toHaveBeenCalled();
});

test('keeps disconnected retrieval in the draft user history without sending the DRES header', async () => {
  const { searchKis: realSearchKis } = jest.requireActual('../../../api/kis');
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
  searchKis.mockImplementation(realSearchKis);
  const fetchSpy = jest.spyOn(global, 'fetch').mockResolvedValue({
    ok: true,
    status: 200,
    json: jest.fn().mockResolvedValue(mockKisResponse({
      inputs: ['a boat crosses the scene'],
      queryText: 'a boat crosses the scene',
      results: [frame],
    })),
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
  expect(searchKis).toHaveBeenCalledWith(expect.objectContaining({ userId: '' }));
  const searchCall = fetchSpy.mock.calls.find(([url]) => String(url).includes('/api/v1/kis/search'));
  expect(searchCall[1].headers['X-VBS-User-ID']).toBeUndefined();

  fireEvent.click(resultImage);
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledWith({
    queryId: createQueryHistory.mock.calls[0][0].queryId,
    frameId: 'disconnected-frame',
  }));
  fetchSpy.mockRestore();
});

test('persists a successful KIS search as a full replay snapshot', async () => {
  const kisResponse = mockKisResponse({
    inputs: ['red boat'],
    queryText: 'red boat',
    events: [{ id: 'E1', text: 'red boat' }],
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
  });
  searchKis.mockResolvedValueOnce(kisResponse);
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('red boat');

  await waitFor(() => expect(createQueryHistory).toHaveBeenCalledWith(expect.objectContaining({
    userId: 'team-a',
    queryText: 'red boat',
    resultSnapshot: expect.objectContaining({
      intent: kisResponse.intent,
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
    }),
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
  searchKis.mockResolvedValueOnce(mockKisResponse({
    inputs: ['red boat'],
    queryText: 'red boat',
    results: [{
      frame_id: 'frame-kis',
      video_id: 'V01',
      frame_idx: 1,
      timestamp_ms: 1000,
      frame_ids: ['frame-kis'],
      timestamps_ms: [1000],
      scores: { final: 0.5 },
    }],
  }));
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('red boat');

  expect(await screen.findByAltText('Frame frame-kis')).toBeTruthy();
  expect(await screen.findByText(/history was not saved/i)).toBeTruthy();
  fireEvent.click(screen.getByAltText('Frame frame-kis'));
  expect(markFrameViewed).not.toHaveBeenCalled();
});

test('keeps the viewed color while allowing a failed activity patch to retry', async () => {
  markFrameViewed.mockRejectedValueOnce(new Error('activity unavailable'));
  searchKis.mockResolvedValueOnce(mockKisResponse({
    inputs: ['red boat'],
    queryText: 'red boat',
    results: [{
      frame_id: 'frame-kis',
      video_id: 'V01',
      frame_idx: 1,
      timestamp_ms: 1000,
      frame_ids: ['frame-kis'],
      timestamps_ms: [1000],
      scores: { final: 0.5 },
    }],
  }));
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('red boat');

  const frameImage = await screen.findByAltText('Frame frame-kis');
  fireEvent.click(frameImage);
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledTimes(1));
  expect(await screen.findByText(/history view state was not recorded/i)).toBeTruthy();
  fireEvent.click(frameImage);
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledTimes(2));
});

test('renders Upload button next to Search and opens file picker', () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });
  const uploadBtn = screen.getByRole('button', { name: 'Upload' });
  expect(uploadBtn).toBeTruthy();

  const fileInput = screen.getByTestId('image-search-file-input');
  expect(fileInput).toBeTruthy();
  const clickSpy = jest.spyOn(fileInput, 'click');
  fireEvent.click(uploadBtn);
  expect(clickSpy).toHaveBeenCalled();
});

test('detects uploaded image and executes image search API on submit', async () => {
  searchFramesByImage.mockResolvedValueOnce({
    results: [{
      frame_id: 'img-result-1',
      video_id: 'V02',
      frame_idx: 10,
      timestamp_ms: 2000,
      frame_ids: ['img-result-1'],
      timestamps_ms: [2000],
      scores: { visual: 0.95 },
    }],
    warnings: [],
    latency: SEARCH_LATENCY,
  });

  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });

  // Initially search is disabled with no query
  const searchBtn = screen.getByRole('button', { name: 'Search' });
  expect(searchBtn.disabled).toBe(true);

  // Upload an image file
  const testFile = new File(['dummy content'], 'query_photo.jpg', { type: 'image/jpeg' });
  const fileInput = screen.getByTestId('image-search-file-input');
  fireEvent.change(fileInput, { target: { files: [testFile] } });

  // Preview badge should now be visible with filename
  expect(await screen.findByText('query_photo.jpg')).toBeTruthy();
  // Search button should now be enabled
  expect(screen.getByRole('button', { name: 'Search' }).disabled).toBe(false);

  // Submit search
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  // It should automatically detect image search and call searchFramesByImage
  await waitFor(() => expect(searchFramesByImage).toHaveBeenCalledWith(
    expect.objectContaining({
      imageFile: testFile,
      topK: 20,
      userId: 'team-a',
    }),
  ));
  expect(searchKis).not.toHaveBeenCalled();

  // Results should render in FramesBox
  expect(await screen.findByAltText('Frame img-result-1')).toBeTruthy();
});

test('clearing the uploaded image restores text query input', async () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });

  const testFile = new File(['dummy'], 'sample.png', { type: 'image/png' });
  const fileInput = screen.getByTestId('image-search-file-input');
  fireEvent.change(fileInput, { target: { files: [testFile] } });

  expect(await screen.findByText('sample.png')).toBeTruthy();

  // Click remove button ✕
  const clearBtn = screen.getByRole('button', { name: 'Remove image' });
  fireEvent.click(clearBtn);

  // Textarea should be restored
  expect(screen.queryByText('sample.png')).toBeNull();
  expect(document.getElementById('event-query')).toBeTruthy();
});

test('New Search resets the uploaded image query', async () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });

  const testFile = new File(['dummy'], 'test.png', { type: 'image/png' });
  const fileInput = screen.getByTestId('image-search-file-input');
  fireEvent.change(fileInput, { target: { files: [testFile] } });

  expect(await screen.findByText('test.png')).toBeTruthy();

  // Click New Search
  fireEvent.click(screen.getByRole('button', { name: 'New Search' }));

  // Image query cleared and textarea restored
  expect(screen.queryByText('test.png')).toBeNull();
  expect(document.getElementById('event-query')).toBeTruthy();
});

test('submitting filter inputs calls filterFrames and displays results in FramesBox', async () => {
  filterFrames.mockResolvedValueOnce({
    results: [
      {
        frame_id: 'filter-res-1',
        video_id: 'Video_01',
        frame_idx: 100,
        timestamp_ms: 5000,
        preview_url: '/frames/filter-res-1.jpg',
      },
    ],
    total_results: 1,
    total_pages: 1,
    page_id: 1,
    page_size: 100,
    latency: 15,
  });

  renderSearch({ topK: 20, setTopK: jest.fn() });

  fireEvent.change(screen.getByPlaceholderText('Folder ID'), { target: { value: 'Folder_01' } });
  fireEvent.change(screen.getByPlaceholderText('Video ID'), { target: { value: 'Video_01' } });
  fireEvent.change(screen.getByPlaceholderText('Title'), { target: { value: 'Traffic' } });
  fireEvent.change(screen.getByPlaceholderText('ASR'), { target: { value: 'vehicle' } });
  fireEvent.change(screen.getByPlaceholderText('OCR'), { target: { value: 'STOP' } });
  fireEvent.change(screen.getByPlaceholderText('Object (name: count)'), { target: { value: 'car: 2, person' } });

  const filterBtn = screen.getByRole('button', { name: 'Filter' });
  expect(filterBtn.disabled).toBe(false);

  fireEvent.click(filterBtn);

  await waitFor(() => expect(filterFrames).toHaveBeenCalledWith(
    expect.objectContaining({
      folderId: 'Folder_01',
      videoId: 'Video_01',
      filters: {
        title: 'Traffic',
        asr: 'vehicle',
        ocr: 'STOP',
        caption: '',
        objects: [
          { value: 'car: 2' },
          { value: 'person: 1' },
        ],
      },
      pageId: 1,
    }),
  ));

  expect(await screen.findByAltText('Frame filter-res-1')).toBeTruthy();
  expect(screen.getByText(/Found/i)).toBeTruthy();
});

test('filter pagination switches pages while reusing active filter parameters', async () => {
  filterFrames.mockResolvedValueOnce({
    results: [
      {
        frame_id: 'filter-page-1',
        video_id: 'Video_01',
        frame_idx: 10,
        timestamp_ms: 1000,
        preview_url: '/frames/filter-page-1.jpg',
      },
    ],
    total_results: 250,
    total_pages: 3,
    page_id: 1,
    pageSize: 100,
    latency: 10,
  });

  renderSearch({ topK: 20, setTopK: jest.fn() });

  fireEvent.change(screen.getByPlaceholderText('Folder ID'), { target: { value: 'Folder_01' } });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));

  expect(await screen.findByRole('navigation', { name: 'Filter result pages' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Page 2' })).toBeTruthy();

  filterFrames.mockResolvedValueOnce({
    results: [
      {
        frame_id: 'filter-page-2',
        video_id: 'Video_01',
        frame_idx: 110,
        timestamp_ms: 11000,
        preview_url: '/frames/filter-page-2.jpg',
      },
    ],
    total_results: 250,
    total_pages: 3,
    page_id: 2,
    pageSize: 100,
    latency: 10,
  });

  const nextBtn = screen.getByRole('button', { name: 'Next page' });
  fireEvent.click(nextBtn);

  await waitFor(() => expect(filterFrames).toHaveBeenCalledWith(
    expect.objectContaining({
      folderId: 'Folder_01',
      pageId: 2,
    }),
  ));

  expect(await screen.findByAltText('Frame filter-page-2')).toBeTruthy();
});

test('Clear button resets all filter fields and disables the Filter button', () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });

  const folderInput = screen.getByPlaceholderText('Folder ID');
  const videoInput = screen.getByPlaceholderText('Video ID');
  const filterBtn = screen.getByRole('button', { name: 'Filter' });
  const clearBtn = screen.getByRole('button', { name: 'Clear' });

  expect(filterBtn.disabled).toBe(true);
  expect(clearBtn.disabled).toBe(true);

  fireEvent.change(folderInput, { target: { value: 'Folder_01' } });
  fireEvent.change(videoInput, { target: { value: 'Video_01' } });

  expect(filterBtn.disabled).toBe(false);
  expect(clearBtn.disabled).toBe(false);

  fireEvent.click(clearBtn);

  expect(folderInput.value).toBe('');
  expect(videoInput.value).toBe('');
  expect(filterBtn.disabled).toBe(true);
  expect(clearBtn.disabled).toBe(true);
});

test('pressing Enter in any filter input field triggers filter submission', async () => {
  filterFrames.mockResolvedValueOnce({
    results: [],
    total_results: 0,
    total_pages: 1,
    page_id: 1,
    pageSize: 100,
    latency: 5,
  });

  renderSearch({ topK: 20, setTopK: jest.fn() });

  const titleInput = screen.getByPlaceholderText('Title');
  fireEvent.change(titleInput, { target: { value: 'Night City' } });
  fireEvent.keyDown(titleInput, { key: 'Enter', code: 'Enter' });

  await waitFor(() => expect(filterFrames).toHaveBeenCalledWith(
    expect.objectContaining({
      filters: expect.objectContaining({ title: 'Night City' }),
      pageId: 1,
    }),
  ));
});
