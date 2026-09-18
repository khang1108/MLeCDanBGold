import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchKis, uploadKisImage } from '../../../api/kis';
import { searchAvs } from '../../../api/avs';
import {
  openQueryHypothesis,
  previewQueryHypothesis,
  commitQueryHypothesis,
  undoQueryHypothesis,
} from '../../../api/queryHypothesis';
import SearchWorkspace, { parseRetrievalDescription } from './SearchWorkspace';
import { filterFrames } from '../../../api/filter';

jest.mock('../../../api/kis', () => ({
  ...jest.requireActual('../../../api/kis'),
  searchKis: jest.fn(),
  uploadKisImage: jest.fn(),
}));
jest.mock('../../../api/avs', () => ({
  searchAvs: jest.fn(),
}));
jest.mock('../../../api/filter', () => ({
  filterFrames: jest.fn(),
}));
jest.mock('../../../api/queryHypothesis', () => ({
  openQueryHypothesis: jest.fn(),
  previewQueryHypothesis: jest.fn(),
  commitQueryHypothesis: jest.fn(),
  undoQueryHypothesis: jest.fn(),
}));
const renderSearch = (props) => render(<SearchWorkspace {...props} />);

beforeEach(() => {
  searchKis.mockReset();
  searchAvs.mockReset();
  uploadKisImage.mockReset();
  filterFrames.mockReset();
  openQueryHypothesis.mockReset();
  previewQueryHypothesis.mockReset();
  commitQueryHypothesis.mockReset();
  undoQueryHypothesis.mockReset();
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

  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    frame: expect.objectContaining({
      video_id: 'L21_a_b.folder2.L21_V001',
      frame_idx: 300,
      fps: 29.97,
    }),
  }));
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

  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    frame: expect.objectContaining({ frame_id: 'frame-degraded' }),
  }));
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



test('keyboard shortcut Ctrl+B toggles KIS chat panel collapse state', () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });
  const chatSidebar = screen.getByRole('complementary', { name: 'KIS search' });
  expect(chatSidebar.classList.contains('collapsed')).toBe(false);

  // Press Ctrl+B to collapse
  fireEvent.keyDown(window, { key: 'b', code: 'KeyB', ctrlKey: true });
  expect(chatSidebar.classList.contains('collapsed')).toBe(true);

  // Press Ctrl+B again to expand
  fireEvent.keyDown(window, { key: 'b', code: 'KeyB', ctrlKey: true });
  expect(chatSidebar.classList.contains('collapsed')).toBe(false);
});

test('keyboard shortcut Ctrl+Alt+B toggles options sidebar collapse state', () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });
  const optionsSidebar = document.querySelector('.adhoc-sidebar');
  expect(optionsSidebar.classList.contains('collapsed')).toBe(false);

  // Press Ctrl+Alt+B to collapse
  fireEvent.keyDown(window, { key: 'b', code: 'KeyB', ctrlKey: true, altKey: true });
  expect(optionsSidebar.classList.contains('collapsed')).toBe(true);

  // Press Ctrl+Alt+B again to expand
  fireEvent.keyDown(window, { key: 'b', code: 'KeyB', ctrlKey: true, altKey: true });
  expect(optionsSidebar.classList.contains('collapsed')).toBe(false);
});

test('keyboard shortcut Ctrl+K focuses chat textarea and auto-expands panel if collapsed', async () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });
  const chatSidebar = screen.getByRole('complementary', { name: 'KIS search' });

  // First collapse the chat panel
  fireEvent.keyDown(window, { key: 'b', code: 'KeyB', ctrlKey: true });
  expect(chatSidebar.classList.contains('collapsed')).toBe(true);

  // Press Ctrl+K
  fireEvent.keyDown(window, { key: 'k', code: 'KeyK', ctrlKey: true });
  expect(chatSidebar.classList.contains('collapsed')).toBe(false);

  await waitFor(() => {
    expect(document.activeElement).toBe(document.getElementById('event-query'));
  });
});

test('keyboard shortcut Ctrl+N resets query and triggers new search', async () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });
  const textarea = document.getElementById('event-query');
  fireEvent.change(textarea, { target: { value: 'some unfinished query' } });
  expect(textarea.value).toBe('some unfinished query');

  // Press Ctrl+N
  fireEvent.keyDown(window, { key: 'n', code: 'KeyN', ctrlKey: true });

  await waitFor(() => {
    expect(document.getElementById('event-query').value).toBe('');
    expect(document.activeElement).toBe(document.getElementById('event-query'));
  });
});

test('keyboard shortcuts Ctrl+1 through Ctrl+6 focus search filter inputs', () => {
  renderSearch({ topK: 20, setTopK: jest.fn() });

  const folderInput = screen.getByPlaceholderText('Folder ID');
  const videoInput = screen.getByPlaceholderText('Video ID');
  const titleInput = screen.getByPlaceholderText('Title');
  const asrInput = screen.getByPlaceholderText('ASR');
  const ocrInput = screen.getByPlaceholderText('OCR');
  const objectInput = screen.getByPlaceholderText('Object (name: count)');

  fireEvent.keyDown(window, { key: '1', code: 'Digit1', ctrlKey: true });
  expect(document.activeElement).toBe(folderInput);

  fireEvent.keyDown(window, { key: '2', code: 'Digit2', ctrlKey: true });
  expect(document.activeElement).toBe(videoInput);

  fireEvent.keyDown(window, { key: '3', code: 'Digit3', ctrlKey: true });
  expect(document.activeElement).toBe(titleInput);

  fireEvent.keyDown(window, { key: '4', code: 'Digit4', ctrlKey: true });
  expect(document.activeElement).toBe(asrInput);

  fireEvent.keyDown(window, { key: '5', code: 'Digit5', ctrlKey: true });
  expect(document.activeElement).toBe(ocrInput);

  fireEvent.keyDown(window, { key: '6', code: 'Digit6', ctrlKey: true });
  expect(document.activeElement).toBe(objectInput);
});

test('submitting a query in AVS mode calls searchAvs and displays AvsHarvestGrid candidates', async () => {
  const mockAvsResults = [
    {
      frame_id: 'avs_f1',
      video_id: 'V100',
      frame_idx: 10,
      timestamp_ms: 1000,
      thumbnail_url: '/thumbs/avs_f1.jpg',
      score: 0.98,
    },
    {
      frame_id: 'avs_f2',
      video_id: 'V101',
      frame_idx: 20,
      timestamp_ms: 2000,
      thumbnail_url: '/thumbs/avs_f2.jpg',
      score: 0.85,
    },
  ];

  searchAvs.mockResolvedValueOnce({
    task_family: 'AVS',
    query: 'red sports car',
    results: mockAvsResults,
    candidate_pool_size: 2,
    unique_videos: 2,
  });

  const onToggleMode = jest.fn();
  renderSearch({
    topK: 80,
    setTopK: jest.fn(),
    workspaceMode: 'AVS',
    onToggleMode,
  });

  // Verify AVS mode indicator in ToolBox
  const avsToggleBtn = screen.getByRole('button', { name: 'AVS' });
  expect(avsToggleBtn.getAttribute('aria-pressed')).toBe('true');

  // Submit AVS query via chat panel input
  const input = document.getElementById('event-query');
  fireEvent.change(input, { target: { value: 'red sports car' } });
  fireEvent.click(screen.getByRole('button', { name: /^(search|update|rewrite)$/i }));

  await waitFor(() => {
    expect(searchAvs).toHaveBeenCalledWith(
      expect.objectContaining({
        query: 'red sports car',
        pageSize: 80,
      }),
    );
  });

  // Verify results summary and selection bar are visible
  expect(await screen.findByText(/showing 2 candidates from 2 videos/i)).toBeTruthy();
  expect(screen.getByLabelText('AVS search candidates')).toBeTruthy();
  expect(screen.getByLabelText('AVS selection bar')).toBeTruthy();
  expect(searchKis).not.toHaveBeenCalled();
});

test('keyboard shortcuts Ctrl+Shift+1 and Ctrl+Shift+2 switch search mode', () => {
  const onToggleMode = jest.fn();
  renderSearch({
    topK: 20,
    setTopK: jest.fn(),
    workspaceMode: 'KIS',
    onToggleMode,
  });

  // Ctrl + Shift + 2 -> AVS
  fireEvent.keyDown(window, { key: '2', code: 'Digit2', ctrlKey: true, shiftKey: true });
  expect(onToggleMode).toHaveBeenCalledWith('AVS');

  // Ctrl + Shift + 1 -> KIS
  fireEvent.keyDown(window, { key: '1', code: 'Digit1', ctrlKey: true, shiftKey: true });
  expect(onToggleMode).toHaveBeenCalledWith('KIS');
});

test('renders HCMUS watermark badge in initial idle state in AVS mode', () => {
  renderSearch({
    workspaceMode: 'AVS',
  });

  expect(screen.getByTestId('hcmus-copyright-badge')).toBeTruthy();
  expect(screen.getAllByText(/MLeCDanBGold · 2026/i).length).toBeGreaterThanOrEqual(1);
});

test('toggling candidate checkbox in AVS mode updates selection and calls onAvsSelectionChange', async () => {
  const onAvsSelectionChange = jest.fn();
  searchAvs.mockResolvedValueOnce({
    results: [
      {
        candidate_id: 'c_1',
        frame_id: 'c_1',
        video_id: 'L21_V001',
        frame_idx: 120,
        timestamp_ms: 4000,
        fps: 30,
        retrieval_rank: 1,
        retrieval_score: 0.95,
        metadata: { frame_idx: 120, timestamp_ms: 4000 },
      },
    ],
    latency: { total_ms: 100, retrieval_ms: 80, coverage_ms: 10, materialization_ms: 10 },
    candidate_pool_size: 1,
    deduplicated_candidate_count: 1,
    unique_videos: 1,
    warnings: [],
  });

  renderSearch({
    workspaceMode: 'AVS',
    onAvsSelectionChange,
  });

  const input = document.getElementById('event-query');
  fireEvent.change(input, { target: { value: 'test candidate' } });
  fireEvent.click(screen.getByRole('button', { name: /^(search|update|rewrite)$/i }));

  await screen.findByText(/showing 1 candidates from 1 videos/i);

  const checkbox = screen.getByRole('checkbox');
  expect(checkbox.checked).toBe(false);

  // Click checkbox to select
  fireEvent.click(checkbox);
  expect(checkbox.checked).toBe(true);
  expect(screen.getByText(/1 selected/i)).toBeTruthy();

  // Click checkbox again to deselect
  fireEvent.click(checkbox);
  expect(checkbox.checked).toBe(false);
  expect(screen.getByText(/0 selected/i)).toBeTruthy();
});

test('opens ImageModal when inspecting a candidate from the AVS review selections drawer', async () => {
  const onFrameClick = jest.fn();
  searchAvs.mockResolvedValueOnce({
    results: [
      {
        candidate_id: 'c_drawer',
        frame_id: 'c_drawer',
        video_id: 'L21_V001',
        frame_idx: 120,
        timestamp_ms: 4000,
        fps: 30,
        retrieval_rank: 1,
        retrieval_score: 0.95,
        metadata: { frame_idx: 120, timestamp_ms: 4000 },
      },
    ],
    latency: { total_ms: 100, retrieval_ms: 80, coverage_ms: 10, materialization_ms: 10 },
    candidate_pool_size: 1,
    deduplicated_candidate_count: 1,
    unique_videos: 1,
    warnings: [],
  });

  renderSearch({
    workspaceMode: 'AVS',
    onFrameClick,
  });

  const input = document.getElementById('event-query');
  fireEvent.change(input, { target: { value: 'inspect drawer test' } });
  fireEvent.click(screen.getByRole('button', { name: /^(search|update|rewrite)$/i }));

  await screen.findByText(/showing 1 candidates from 1 videos/i);

  // Select the candidate
  fireEvent.click(screen.getByRole('checkbox'));
  expect(screen.getByText(/1 selected/i)).toBeTruthy();

  // Open Review selections drawer
  fireEvent.click(screen.getByRole('button', { name: 'Review selections' }));
  expect(screen.getByRole('dialog', { name: 'Selected AVS answers' })).toBeTruthy();

  // Click candidate inside drawer to inspect
  const drawerItem = screen.getByRole('button', { name: 'Inspect candidate c_drawer' });
  fireEvent.click(drawerItem);

  expect(onFrameClick).toHaveBeenCalledTimes(1);
  expect(onFrameClick).toHaveBeenCalledWith(
    expect.objectContaining({
      frame: expect.objectContaining({ candidate_id: 'c_drawer' }),
    }),
  );
});

test('asserts a stale-results notice appears after a committed query edit and disappears after a new search at that revision', async () => {
  searchKis.mockResolvedValueOnce(mockKisResponse({
    queryText: 'person walking',
    revision: 1,
    results: [frameResult('frame-1')],
  }));

  renderSearch({ topK: 20, setTopK: jest.fn() });
  submit('person walking');

  expect(await screen.findByAltText('Frame frame-1')).toBeTruthy();
  expect(screen.queryByTestId('stale-results-notice')).toBeNull();

  previewQueryHypothesis.mockResolvedValueOnce({
    base_revision: 1,
    intent: {
      revision: 2,
      query_text: 'person walking slowly',
      events: [{ id: 'E1', text: 'person walking slowly', origin: 'user_override' }],
    },
  });

  commitQueryHypothesis.mockResolvedValueOnce({
    session_id: 'qh_snap_1',
    query_revision: 2,
    intent: {
      revision: 2,
      query_text: 'person walking slowly',
      events: [{ id: 'E1', text: 'person walking slowly', origin: 'user_override' }],
    },
    can_undo: true,
  });

  fireEvent.click(screen.getByTitle('Edit event text'));
  const editInput = screen.getByDisplayValue('test');
  fireEvent.change(editInput, { target: { value: 'person walking slowly' } });
  fireEvent.click(screen.getByRole('button', { name: /preview edit/i }));

  const applyBtn = await screen.findByRole('button', { name: /apply/i });
  fireEvent.click(applyBtn);

  const staleNotices = await screen.findAllByTestId('stale-results-notice');
  expect(staleNotices.length).toBeGreaterThanOrEqual(1);
  expect(screen.getByAltText('Frame frame-1')).toBeTruthy();

  searchKis.mockResolvedValueOnce(mockKisResponse({
    queryText: 'person walking slowly',
    revision: 2,
    results: [frameResult('frame-2')],
  }));

  const updateBtn = screen.getByRole('button', { name: /update results/i });
  fireEvent.click(updateBtn);

  await waitFor(() => expect(searchKis).toHaveBeenCalledWith(
    expect.objectContaining({
      operation: { kind: 'search_only' },
      expectedRevision: 2,
    }),
  ));

  expect(await screen.findByAltText('Frame frame-2')).toBeTruthy();
  expect(screen.queryAllByTestId('stale-results-notice')).toHaveLength(0);
});





