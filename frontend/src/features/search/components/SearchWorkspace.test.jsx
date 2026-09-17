import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchKis, uploadKisImage } from '../../../api/kis';
import SearchWorkspace, { parseRetrievalDescription } from './SearchWorkspace';
import { filterFrames } from '../../../api/filter';
import {
  createQueryHistory,
  markFrameViewed,
  recordQueryInteraction,
} from '../../../api/history';

jest.mock('../../../api/kis', () => ({
  ...jest.requireActual('../../../api/kis'),
  searchKis: jest.fn(),
  uploadKisImage: jest.fn(),
}));
jest.mock('../../../api/filter', () => ({
  filterFrames: jest.fn(),
}));
jest.mock('../../../api/history', () => ({
  createQueryHistory: jest.fn(),
  markFrameViewed: jest.fn(),
  recordQueryInteraction: jest.fn(),
}));
const renderSearch = (props) => render(<SearchWorkspace {...props} />);

beforeEach(() => {
  searchKis.mockReset();
  uploadKisImage.mockReset();
  filterFrames.mockReset();
  createQueryHistory.mockResolvedValue({});
  markFrameViewed.mockResolvedValue({});
  recordQueryInteraction.mockResolvedValue({});
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
  operationSummary = { kind: 'initial_resolve', affected_event_ids: ['E1'] },
  evidenceSnapshotId = 'snap_1',
} = {}) => ({
  intent: {
    revision,
    inputs,
    query_text: queryText,
    entities: [],
    events,
    temporal_edges: [],
  },
  evidence_snapshot_id: evidenceSnapshotId,
  operation_summary: operationSummary,
  results,
  warnings,
  latency,
});

const submit = (eventDescription) => {
  fireEvent.change(document.getElementById('event-query'), {
    target: { value: eventDescription },
  });
  fireEvent.click(screen.getByRole('button', { name: /^(search|update|rewrite)$/i }));
};

test.each([
  ['a red vehicle passes', { kind: 'initial_resolve', text: 'a red vehicle passes' }],
  ['E1: a person enters the room', { kind: 'initial_resolve', patches: [expect.objectContaining({ event_id: 'E1', instruction: 'a person enters the room' })] }],
])('routes %s through frame search', async (
  description,
  expectedOp,
) => {
  searchKis.mockResolvedValueOnce(mockKisResponse({ inputs: [description], queryText: description }));
  renderSearch({ topK: 20, setTopK: jest.fn() });
  submit(description);

  await waitFor(() => expect(searchKis).toHaveBeenCalledWith(
    expect.objectContaining({ operation: expect.objectContaining(expectedOp), topK: 20 }),
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
    expect.objectContaining({
      operation: expect.objectContaining({
        kind: 'initial_resolve',
        patches: [expect.objectContaining({ event_id: 'E1', instruction: 'a person enters the room' })],
      }),
      topK: 20,
    }),
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
      operation: expect.objectContaining({ kind: 'initial_resolve', text: 'a lexical-only query' }),
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

test('hands off exact live EventTrail context when opening a result', async () => {
  const onFrameClick = jest.fn();
  const response = mockKisResponse({
    inputs: ['woman enters'],
    queryText: 'woman enters',
    revision: 3,
    events: [
      { id: 'E1', text: 'woman enters', images: [] },
      { id: 'E2', text: 'woman sits', images: [] },
    ],
    results: [{
      result_id: 'r_1',
      frame_id: 'frame-explore',
      video_id: 'V01',
      frame_idx: 125,
      timestamp_ms: 10_010,
      fps: 29.97,
      caption: 'woman enters',
      scores: { final: 0.91 },
    }],
    evidenceSnapshotId: 'snap_1',
  });
  searchKis.mockResolvedValueOnce(response);
  renderSearch({ topK: 20, setTopK: jest.fn(), onFrameClick, userId: 'team-a' });
  submit('woman enters');

  fireEvent.click(await screen.findByAltText('Frame frame-explore'));

  expect(onFrameClick).toHaveBeenCalledWith({
    frame: expect.objectContaining({ result_id: 'r_1', frame_id: 'frame-explore', video_id: 'V01' }),
    eventTrailContext: {
      snapshotId: 'snap_1',
      resultId: 'r_1',
      kisRevision: 3,
      events: [
        { id: 'E1', text: 'woman enters', images: [] },
        { id: 'E2', text: 'woman sits', images: [] },
      ],
      searchSessionId: expect.anything(),
    },
  });
});

test('omits eventTrailContext when evidence_snapshot_id is null or when in filter mode', async () => {
  const onFrameClick = jest.fn();
  const response = mockKisResponse({
    inputs: ['red boat'],
    queryText: 'red boat',
    results: [{
      result_id: 'r_1',
      frame_id: 'frame-degraded',
      video_id: 'V01',
      frame_idx: 100,
      timestamp_ms: 10_010,
      score: 0.9,
    }],
    evidenceSnapshotId: null,
  });
  searchKis.mockResolvedValueOnce(response);
  renderSearch({ topK: 20, setTopK: jest.fn(), onFrameClick });
  submit('red boat');

  fireEvent.click(await screen.findByAltText('Frame frame-degraded'));

  expect(onFrameClick).toHaveBeenCalledWith({
    frame: expect.objectContaining({ frame_id: 'frame-degraded' }),
  });
});

test('calls onEventTrailInvalidated when a new search commits results', async () => {
  const onEventTrailInvalidated = jest.fn();
  searchKis.mockResolvedValueOnce(mockKisResponse({
    inputs: ['query 1'],
    queryText: 'query 1',
    results: [{ frame_id: 'f1', video_id: 'V01', frame_idx: 100, timestamp_ms: 1000, score: 0.9 }],
  }));

  renderSearch({ topK: 20, setTopK: jest.fn(), onEventTrailInvalidated });
  submit('query 1');

  await screen.findByAltText('Frame f1');
  expect(onEventTrailInvalidated).toHaveBeenCalledTimes(1);
});

const frameResult = (id) => ({
  frame_id: id,
  video_id: 'V01',
  frame_idx: id === 'frame-1' ? 100 : 200,
  timestamp_ms: id === 'frame-1' ? 4_000 : 8_000,
  fps: 25,
  caption: id,
  score: 0.9,
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

  submit('E2: second');
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

  submit('E2: second clue');
  expect(await screen.findByText('Network error (500)')).toBeTruthy();
  expect(screen.getByAltText('Frame frame-1')).toBeTruthy();
  const input = document.getElementById('event-query');
  expect(input.value).toBe('E2: second clue');
});

test('preserves committed warnings while the next revision is pending and fails', async () => {
  let rejectSecond;
  searchKis
    .mockResolvedValueOnce(mockKisResponse({
      queryText: 'first',
      revision: 1,
      warnings: ['first response warning'],
      results: [frameResult('frame-1')],
    }))
    .mockImplementationOnce(() => new Promise((resolve, reject) => {
      rejectSecond = reject;
    }));

  renderSearch({ topK: 20, setTopK: jest.fn() });
  submit('first');
  expect(await screen.findByText('first response warning')).toBeTruthy();

  submit('E2: second clue');
  expect(screen.getByText('first response warning')).toBeTruthy();

  rejectSecond(new Error('second request failed'));
  expect(await screen.findByText('second request failed')).toBeTruthy();
  expect(screen.getByText('first response warning')).toBeTruthy();
  expect(screen.getByAltText('Frame frame-1')).toBeTruthy();
});

test('opens a KIS result exactly once and records one viewed-frame write', async () => {
  const onFrameClick = jest.fn();
  const response = {
    ...mockKisResponse({
      queryText: 'red boat',
      revision: 1,
      results: [{
        result_id: 'r_1',
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
    eventTrailContext: expect.any(Object),
  }));
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledTimes(1));
  expect(markFrameViewed).toHaveBeenCalledWith(expect.objectContaining({
    frameId: 'frame-explore',
  }));
});

test('a frame clicked before createQueryHistory resolves does not call markFrameViewed or recordQueryInteraction until creation resolves', async () => {
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
  expect(recordQueryInteraction).not.toHaveBeenCalled();

  resolveHistory({});
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(recordQueryInteraction).toHaveBeenCalledTimes(1));
  expect(markFrameViewed).toHaveBeenCalledWith(expect.objectContaining({
    frameId: 'frame-1',
  }));
  expect(recordQueryInteraction).toHaveBeenCalledWith(expect.objectContaining({
    eventType: 'result_open',
    frameId: 'frame-1',
  }));
});

test('opening submission from an active query enqueues submission interaction event with current revision', async () => {
  const onOpenSubmission = jest.fn();
  searchKis.mockResolvedValueOnce(mockKisResponse({
    queryText: 'first',
    revision: 2,
    results: [frameResult('frame-1')],
  }));

  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a', onOpenSubmission });
  submit('first');
  const submitButton = await screen.findByRole('button', { name: /submit this frame to dres/i });
  fireEvent.click(submitButton);

  expect(onOpenSubmission).toHaveBeenCalledWith(expect.objectContaining({
    videoId: 'V01',
    startMs: 4000,
  }));
  await waitFor(() => expect(recordQueryInteraction).toHaveBeenCalledWith(expect.objectContaining({
    eventType: 'submission',
    semanticRevision: 2,
    videoId: 'V01',
  })));
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

test('skips a queued viewed-frame write when history creation aborts', async () => {
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

  const abortError = new Error('history request aborted');
  abortError.name = 'AbortError';
  await act(async () => {
    rejectHistory(abortError);
    await Promise.resolve();
    await Promise.resolve();
  });

  expect(markFrameViewed).not.toHaveBeenCalled();
  expect(screen.queryByText(/history was not saved/i)).toBeNull();
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

  submit('E2: second');
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
  submit('E2: second');
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
    expect.objectContaining({
      operation: expect.objectContaining({ kind: 'initial_resolve', text: 'a red vehicle passes' }),
      userId: '',
    }),
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
      score: 0.5,
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

test('attaches image via composer and executes multimodal KIS search on submit', async () => {
  uploadKisImage.mockResolvedValueOnce({
    asset_id: 'ast_test_1',
    file_name: 'query_photo.jpg',
  });
  searchKis.mockResolvedValueOnce(mockKisResponse({
    results: [{
      frame_id: 'img-result-1',
      video_id: 'V02',
      frame_idx: 10,
      timestamp_ms: 2000,
      frame_ids: ['img-result-1'],
      timestamps_ms: [2000],
      score: 0.95,
      scores: { visual: 0.95 },
    }],
  }));

  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });

  // Attach an image file via the unified composer
  const testFile = new File(['dummy content'], 'query_photo.jpg', { type: 'image/jpeg' });
  const fileInput = screen.getByLabelText(/attach image/i);
  fireEvent.change(fileInput, { target: { files: [testFile] } });

  await waitFor(() => expect(uploadKisImage).toHaveBeenCalledWith(
    expect.objectContaining({ imageFile: testFile }),
  ));

  const searchBtn = await screen.findByRole('button', { name: 'Search' });
  await waitFor(() => expect(searchBtn.disabled).toBe(false));

  // Submit search
  fireEvent.click(searchBtn);

  await waitFor(() => expect(searchKis).toHaveBeenCalledWith(
    expect.objectContaining({
      operation: expect.objectContaining({
        kind: 'initial_resolve',
        image_refs: expect.arrayContaining([expect.objectContaining({ asset_id: 'ast_test_1' })]),
      }),
      topK: 20,
      userId: 'team-a',
    }),
  ));

  expect(await screen.findByAltText('Frame img-result-1')).toBeTruthy();
});

test('Search-only rerun when only retrieval controls change with active intent', async () => {
  const initialResponse = mockKisResponse({
    queryText: 'red boat',
    events: [{ id: 'E1', text: 'red boat' }],
    revision: 1,
    results: [{
      frame_id: 'boat-1',
      video_id: 'V01',
      frame_idx: 10,
      timestamp_ms: 1000,
      frame_ids: ['boat-1'],
      timestamps_ms: [1000],
    }],
  });
  searchKis.mockResolvedValue(initialResponse);

  const { rerender } = renderSearch({ topK: 20, setTopK: jest.fn() });
  submit('red boat');

  await waitFor(() => expect(searchKis).toHaveBeenCalledTimes(1));

  // Rerender with changed topK without a draft
  rerender(<SearchWorkspace topK={50} setTopK={jest.fn()} />);

  await waitFor(() => expect(searchKis).toHaveBeenCalledWith(
    expect.objectContaining({
      operation: { kind: 'search_only' },
      topK: 50,
      expectedRevision: 1,
    }),
  ));
});

test('New Search resets KIS session', async () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });

  const textarea = document.getElementById('event-query');
  fireEvent.change(textarea, { target: { value: 'test draft' } });
  expect(textarea.value).toBe('test draft');

  // Click New Search
  fireEvent.click(screen.getByRole('button', { name: 'New Search' }));

  expect(document.getElementById('event-query').value).toBe('');
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

test('shows gif loader in place of welcome empty state while search is in progress', async () => {
  let resolveSearch;
  searchKis.mockImplementationOnce(() => new Promise((resolve) => { resolveSearch = resolve; }));

  renderSearch({ topK: 20, setTopK: jest.fn() });
  expect(screen.getByTestId('hcmus-copyright-badge')).toBeTruthy();
  expect(screen.queryByTestId('gif-loader')).toBeNull();

  submit('test search');

  expect(await screen.findByTestId('gif-loader')).toBeTruthy();
  expect(screen.queryByTestId('hcmus-copyright-badge')).toBeNull();

  resolveSearch(mockKisResponse({ queryText: 'test search', revision: 1, results: [frameResult('frame-1')] }));
  expect(await screen.findByAltText('Frame frame-1')).toBeTruthy();
  expect(screen.queryByTestId('gif-loader')).toBeNull();
});

test('Step 1 & 2 (Task 7): preserves ranked result order and shows Trail annotations without reranking or hiding results', async () => {
  const response = mockKisResponse({
    queryText: 'test order preservation',
    evidenceSnapshotId: 'snap_order',
    results: [
      { result_id: 'r_1', frame_id: 'f1', video_id: 'V01', frame_idx: 10, timestamp_ms: 1000, score: 0.95 },
      { result_id: 'r_2', frame_id: 'f2', video_id: 'V02', frame_idx: 20, timestamp_ms: 2000, score: 0.85 },
    ],
  });
  searchKis.mockResolvedValueOnce(response);

  renderSearch({
    topK: 20,
    setTopK: jest.fn(),
    eventTrailAnnotations: {
      'snap_order:r_1': 'exhausted',
      'snap_order:r_2': 'explored',
    },
  });

  submit('test order preservation');

  expect(await screen.findByAltText('Frame f1')).toBeTruthy();
  expect(await screen.findByAltText('Frame f2')).toBeTruthy();

  // Annotations are rendered
  expect(screen.getByText('Exhausted')).toBeTruthy();
  expect(screen.getByText('Explored')).toBeTruthy();

  // The order is preserved: f1 is first, f2 is second
  const images = screen.getAllByRole('img');
  const frameImages = images.filter((img) => img.alt && img.alt.startsWith('Frame '));
  expect(frameImages[0].getAttribute('alt')).toBe('Frame f1');
  expect(frameImages[1].getAttribute('alt')).toBe('Frame f2');

  // searchKis called only once for the search, not again for annotations
  expect(searchKis).toHaveBeenCalledTimes(1);
});

test('Step 2 (Task 8): EventTrail context passes searchSessionId only when real history query session exists', async () => {
  const onFrameClick = jest.fn();
  const response = mockKisResponse({
    queryText: 'query correlation test',
    evidenceSnapshotId: 'snap_corr',
    results: [{
      result_id: 'r_1',
      frame_id: 'f1',
      video_id: 'V01',
      frame_idx: 10,
      timestamp_ms: 1000,
      score: 0.9,
    }],
  });
  searchKis.mockResolvedValueOnce(response);

  // Without userId -> searchSessionId must be null, not a generated surrogate
  const { unmount } = renderSearch({ topK: 20, setTopK: jest.fn(), onFrameClick, userId: '' });
  submit('query correlation test');

  fireEvent.click(await screen.findByAltText('Frame f1'));
  expect(onFrameClick).toHaveBeenCalledWith(
    expect.objectContaining({
      eventTrailContext: expect.objectContaining({
        snapshotId: 'snap_corr',
        searchSessionId: null,
      }),
    }),
  );

  unmount();

  // With userId -> searchSessionId must be the real query ID string
  onFrameClick.mockReset();
  searchKis.mockResolvedValueOnce(response);
  renderSearch({ topK: 20, setTopK: jest.fn(), onFrameClick, userId: 'team-a' });
  submit('query correlation test');

  fireEvent.click(await screen.findByAltText('Frame f1'));
  expect(onFrameClick).toHaveBeenCalledWith(
    expect.objectContaining({
      eventTrailContext: expect.objectContaining({
        snapshotId: 'snap_corr',
        searchSessionId: expect.stringMatching(/^query-/),
      }),
    }),
  );
});



