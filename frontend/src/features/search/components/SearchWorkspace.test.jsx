import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFrames, searchTrake } from '../../../api/search';
import SearchWorkspace, {
  parseRetrievalDescription,
  parseTrakeEvents,
} from './SearchWorkspace';
import { SubmissionProvider } from '../../submission/contexts/SubmissionContext';
import { SubmissionDialogProvider } from '../../submission/contexts/SubmissionDialogContext';
import {
  createQueryHistory,
  getSubmissionFiles,
  markFrameViewed,
} from '../../../api/workspace';

jest.mock('../../../api/search');
jest.mock('../../../api/workspace', () => ({
  getSubmissionFiles: jest.fn(),
  workspaceWebSocketUrl: jest.fn(() => 'ws://example.test/api/v1/workspace/ws'),
  createQueryHistory: jest.fn(),
  markFrameViewed: jest.fn(),
  markFramesSubmitted: jest.fn(),
}));

class MockWebSocket {
  static instances = [];
  static OPEN = 1;
  constructor() { this.readyState = 0; this.sent = []; MockWebSocket.instances.push(this); }
  send(value) { this.sent.push(value); }
  open() { this.readyState = 1; this.onopen?.(); }
  message(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }
  close() { this.readyState = 3; this.onclose?.(); }
}

const renderSearch = (props) => {
  const result = render(
    <SubmissionProvider>
      <SubmissionDialogProvider><SearchWorkspace {...props} /></SubmissionDialogProvider>
    </SubmissionProvider>,
  );
  act(() => MockWebSocket.instances[MockWebSocket.instances.length - 1].open());
  return result;
};

beforeEach(() => {
  searchFrames.mockReset();
  searchTrake.mockReset();
  // Search behavior does not need the shared file list. Leaving hydration
  // pending keeps this suite focused and avoids unrelated async state updates.
  getSubmissionFiles.mockReturnValue(new Promise(() => {}));
  createQueryHistory.mockResolvedValue({});
  markFrameViewed.mockResolvedValue({});
  MockWebSocket.instances = [];
  window.WebSocket = MockWebSocket;
});

afterEach(() => {
  delete window.WebSocket;
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
  expect(parseTrakeEvents('A video description. E1: only one event')).toEqual(['only one event']);
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
  renderSearch({ topK: 20, setTopK: jest.fn() });
  submit(description);

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query, topK: 20 }),
  ));
  expect(searchTrake).not.toHaveBeenCalled();
  expect(await screen.findByText('No frames found matching your query')).toBeTruthy();
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
      },
      {
        video_id: 'V01',
        score: 2.8,
        frame_ids: ['b1', 'b2'],
        frame_idxs: [30, 40],
        timestamps_ms: [3000, 4000],
      },
    ],
    total_results: 3,
    warnings: [],
    latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn() });
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

test('TRAKE accepts one labeled event and opens its frame as read-only', async () => {
  const onFrameClick = jest.fn();
  searchTrake.mockResolvedValueOnce({
    events: ['only one event'],
    paths: [{
      video_id: 'V01',
      score: 1.0,
      frame_ids: ['f1'],
      frame_idxs: [10],
      timestamps_ms: [1000],
    }],
    warnings: [],
    latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn(), onFrameClick });
  submit('A short video. E1: only one event');

  await waitFor(() => expect(searchTrake).toHaveBeenCalledWith(
    expect.objectContaining({ events: ['only one event'], topK: 20 }),
  ));
  fireEvent.click(await screen.findByRole('button', { name: /view event E1/i }));

  expect(onFrameClick).toHaveBeenCalledWith({
    frame: expect.objectContaining({ frame_id: 'f1', frame_idx: 10 }),
    submissionMode: 'none',
  });
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
    submissionMode: 'kis',
  });
});

test('does not render the retired query-helper control', () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });
  expect(screen.queryByRole('button', { name: /suggest query/i })).toBeNull();
});

test('requires the shared User ID before starting live retrieval', async () => {
  const onFocusUserId = jest.fn();
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: '  ', onFocusUserId });
  submit('a red vehicle passes');

  expect((await screen.findByRole('alert')).textContent).toMatch(/user id/i);
  expect(onFocusUserId).toHaveBeenCalledTimes(1);
  expect(searchFrames).not.toHaveBeenCalled();
  expect(createQueryHistory).not.toHaveBeenCalled();
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

test('persists a successful TRAKE search while preserving path order', async () => {
  searchTrake.mockResolvedValueOnce({
    events: ['first event', 'second event'],
    paths: [{
      video_id: 'V01',
      score: 3.0,
      frame_ids: ['f2', 'f1'],
      frame_idxs: [20, 10],
      timestamps_ms: [2000, 1000],
    }],
    warnings: [],
    latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('A video. E1: first event E2: second event');

  await waitFor(() => expect(createQueryHistory).toHaveBeenCalled());
  expect(createQueryHistory.mock.calls[0][0].resultSnapshot).toEqual({
    events: ['first event', 'second event'],
    latency: SEARCH_LATENCY,
    warnings: [],
    paths: [{
      video_id: 'V01',
      score: 3,
      frame_ids: ['f2', 'f1'],
      frame_idxs: [20, 10],
      timestamps_ms: [2000, 1000],
    }],
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

test('highlights a frame only after its active query records the submission', async () => {
  searchFrames.mockResolvedValueOnce({
    results: [{
      frame_id: 'frame-kis',
      video_id: 'V01',
      frame_idx: 125,
      timestamp_ms: 1000,
      frame_ids: ['frame-kis'],
      timestamps_ms: [1000],
      scores: { final: 0.8 },
      caption: 'A test frame',
    }],
    warnings: [],
    latency: SEARCH_LATENCY,
  });
  renderSearch({ topK: 20, setTopK: jest.fn(), userId: 'team-a' });
  submit('red boat');

  const submitBtn = await screen.findByRole('button', { name: /submit this frame/i });
  await waitFor(() => expect(createQueryHistory).toHaveBeenCalledTimes(1));
  const queryId = createQueryHistory.mock.calls[0][0].queryId;
  const card = screen.getByAltText('Frame frame-kis').closest('.frame-card');

  fireEvent.click(screen.getByAltText('Frame frame-kis'));
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledWith({ queryId, frameId: 'frame-kis' }));
  expect(card.classList.contains('viewed')).toBe(true);

  fireEvent.click(submitBtn);
  expect(card.classList.contains('submitted')).toBe(false);

  await act(async () => window.dispatchEvent(new CustomEvent('hcmai:history-changed', {
    detail: { queryId: 'another-query', frameIds: ['frame-kis'] },
  })));
  expect(card.classList.contains('submitted')).toBe(false);

  await act(async () => window.dispatchEvent(new CustomEvent('hcmai:history-changed', {
    detail: { queryId, frameIds: ['frame-kis'] },
  })));

  const submittedBtn = await screen.findByRole('button', { name: /frame submitted/i });
  expect(submittedBtn.textContent).toBe('✓ Submitted');
  expect(card.classList.contains('submitted')).toBe(true);
});
