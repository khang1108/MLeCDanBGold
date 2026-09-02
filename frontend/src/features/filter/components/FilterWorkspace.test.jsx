import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import FilterWorkspace from './FilterWorkspace';
import { SubmissionProvider } from '../../submission/contexts/SubmissionContext';
import { filterFrames } from '../../../api/filter';
import { getFrameDetail } from '../../../api/frames';

jest.mock('../../../api/filter');
jest.mock('../../../api/frames');

const renderWorkspace = (props = {}) => render(
  <SubmissionProvider>
    <FilterWorkspace {...props} />
  </SubmissionProvider>,
);

beforeEach(() => {
  filterFrames.mockReset();
  getFrameDetail.mockReset();
  getFrameDetail.mockImplementation(async ({ frameId }) => ({
    frame_id: frameId,
    frame_url: `http://example.test/${frameId}.jpg`,
  }));
});

test('renders every available folder', () => {
  renderWorkspace({ isActive: false });

  expect(screen.getByLabelText('Folder')).toBeTruthy();
  fireEvent.focus(screen.getByLabelText('Folder'));
  expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual([
    'L21', 'L22', 'L23', 'L24', 'L25',
    'L26', 'L27', 'L28', 'L29', 'L30',
  ]);
});

test('filters folder options as the user types', () => {
  renderWorkspace({ isActive: false });

  const folderInput = screen.getByLabelText('Folder');
  fireEvent.change(folderInput, { target: { value: 'L26' } });

  expect(screen.getByRole('option', { name: 'L26' })).toBeTruthy();
  expect(screen.queryByRole('option', { name: 'L21' })).toBeNull();
});

test('switches video scope to result-backed typeahead options after filtering', async () => {
  filterFrames.mockResolvedValueOnce({
    total_pages: 1,
    results: [
      {
        frame_id: 'frame-a',
        video_id: 'L26_topic.video-1',
        folder_id: 'L26',
        frame_idx: 10,
        timestamp_ms: 1000,
      },
      {
        frame_id: 'frame-b',
        video_id: 'L26_topic.video-2',
        folder_id: 'L26',
        frame_idx: 20,
        timestamp_ms: 2000,
      },
    ],
  });
  renderWorkspace({ isActive: false });

  expect(screen.getByLabelText('Video').getAttribute('role')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));
  await waitFor(() => expect(screen.getByLabelText('Video').getAttribute('role'))
    .toBe('combobox'));

  const videoInput = screen.getByLabelText('Video');
  fireEvent.change(videoInput, { target: { value: 'video-2' } });
  expect(screen.getByRole('option', { name: 'L26_topic.video-2' })).toBeTruthy();
  expect(screen.queryByRole('option', { name: 'L26_topic.video-1' })).toBeNull();
});

test('shows the same welcome copy as the query page before filtering', () => {
  renderWorkspace({ isActive: false });

  expect(screen.getByText('Welcome to HCMAI Frame Search')).toBeTruthy();
  expect(screen.getByText(
    'Enter a natural language question or keywords above to query the video corpus.',
  )).toBeTruthy();
});

test('keeps object controls compact and supports adding rows', () => {
  renderWorkspace({ isActive: false });

  const objectInput = screen.getByLabelText('Object 1, format name colon count');
  fireEvent.change(objectInput, { target: { value: 'chair: 4' } });
  fireEvent.click(screen.getByRole('button', { name: 'Add object filter' }));

  expect(screen.getByDisplayValue('chair: 4')).toBeTruthy();
  expect(screen.getByLabelText('Object 2, format name colon count')).toBeTruthy();
});

test('uses vertically growing text fields for long filter values', () => {
  renderWorkspace({ isActive: false });

  expect(screen.getByLabelText('Title').tagName).toBe('TEXTAREA');
  expect(screen.getByLabelText('ASR / Transcript').tagName).toBe('TEXTAREA');
});

test('renders the backend page without client-side rescoping', async () => {
  filterFrames.mockResolvedValueOnce({
    total_pages: 1,
    results: [
      {
        rank: 1,
        frame_id: 'frame-a',
        video_id: 'L26_topic.video-1',
        folder_id: 'L26',
        frame_idx: 10,
        timestamp_ms: 1000,
        caption: 'Topic A',
      },
      {
        rank: 2,
        frame_id: 'frame-b',
        video_id: 'L27_topic.video-1',
        folder_id: 'L27',
        frame_idx: 20,
        timestamp_ms: 2000,
        caption: 'Topic B',
      },
    ],
  });
  renderWorkspace({ isActive: false });

  fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'Boat' } });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));

  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(1));
  expect(filterFrames).toHaveBeenCalledWith(expect.objectContaining({
    filters: expect.objectContaining({ title: 'Boat' }),
    pageId: 1,
    framesPerPage: expect.any(Number),
  }));
  expect((await screen.findAllByText('Topic A')).length).toBeGreaterThan(0);
  expect((screen.getAllByText('Topic B')).length).toBeGreaterThan(0);
  expect(document.querySelectorAll('.filter-video-group')).toHaveLength(0);

  fireEvent.change(screen.getByLabelText('Folder'), { target: { value: 'L26' } });
  expect(await screen.findByAltText('Frame frame-a')).toBeTruthy();
  expect(await screen.findByAltText('Frame frame-b')).toBeTruthy();

  fireEvent.change(screen.getByLabelText('Video'), {
    target: { value: 'L26_topic.video-1' },
  });
  expect(await screen.findByAltText('Frame frame-a')).toBeTruthy();
  expect(filterFrames).toHaveBeenCalledTimes(1);
});

test('includes the current folder and video scope in the Filter request', async () => {
  filterFrames.mockResolvedValueOnce({ results: [] });
  renderWorkspace({ isActive: false });

  fireEvent.change(screen.getByLabelText('Folder'), { target: { value: 'L26' } });
  fireEvent.change(screen.getByLabelText('Video'), {
    target: { value: '' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));

  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(1));
  const [request] = filterFrames.mock.calls[0];
  expect(request).toEqual(expect.objectContaining({
    folderId: 'L26',
    videoId: '',
  }));
});

test('uses applied scope for pagination without filtering the backend page again', async () => {
  filterFrames
    .mockResolvedValueOnce({
      total_pages: 2,
      results: [
        {
          frame_id: 'frame-a',
          video_id: 'L26_topic.video-1',
          folder_id: 'L26',
          frame_idx: 1,
          timestamp_ms: 40,
        },
        {
          frame_id: 'frame-b',
          video_id: 'L27_topic.video-1',
          folder_id: 'L27',
          frame_idx: 2,
          timestamp_ms: 80,
        },
      ],
    })
    .mockResolvedValueOnce({
      total_pages: 2,
      results: [
        {
          frame_id: 'frame-c',
          video_id: 'L26_topic.video-2',
          folder_id: 'L26',
          frame_idx: 3,
          timestamp_ms: 120,
        },
        {
          frame_id: 'frame-d',
          video_id: 'L27_topic.video-2',
          folder_id: 'L27',
          frame_idx: 4,
          timestamp_ms: 160,
        },
      ],
    });

  renderWorkspace({ isActive: false });
  fireEvent.change(screen.getByLabelText('Folder'), { target: { value: 'L26' } });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));
  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(1));
  expect(filterFrames.mock.calls[0][0]).toEqual(expect.objectContaining({
    folderId: 'L26',
    pageId: 1,
  }));
  expect(await screen.findByAltText('Frame frame-a')).toBeTruthy();
  expect(await screen.findByAltText('Frame frame-b')).toBeTruthy();

  // This scope remains a draft until Filter is pressed again.
  fireEvent.change(screen.getByLabelText('Folder'), { target: { value: 'L27' } });
  expect(await screen.findByAltText('Frame frame-b')).toBeTruthy();
  expect(await screen.findByAltText('Frame frame-a')).toBeTruthy();

  fireEvent.click(screen.getByRole('button', { name: 'Page 2' }));
  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(2));
  expect(filterFrames.mock.calls[1][0]).toEqual(expect.objectContaining({
    folderId: 'L26',
    pageId: 2,
  }));
  expect(await screen.findByAltText('Frame frame-d')).toBeTruthy();
  expect(await screen.findByAltText('Frame frame-c')).toBeTruthy();
});

test('uses complete Filter rows without requesting per-frame details', async () => {
  const onFrameClick = jest.fn();
  filterFrames.mockResolvedValueOnce({
    total_pages: 1,
    results: Array.from({ length: 12 }, (_, index) => ({
      frame_id: `frame-${index}`,
      video_id: 'L21_V001',
      folder_id: 'L21',
      frame_idx: index,
      timestamp_ms: index * 40,
      title: 'Episode',
      caption: `Complete caption ${index}`,
      ocr: null,
      objects: { person: 3 },
      asr: 'Xin chào',
    })),
  });
  renderWorkspace({ isActive: false, onFrameClick });

  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));
  const firstFrame = await screen.findByAltText('Frame frame-0');
  await waitFor(() => expect(document.querySelectorAll('.frame-card')).toHaveLength(12));
  expect(getFrameDetail).not.toHaveBeenCalled();

  fireEvent.click(firstFrame);
  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    frame_id: 'frame-0',
    caption: 'Complete caption 0',
    objects: { person: 3 },
    asr: 'Xin chào',
  }));
});

test('requests page 1 for a new filter after navigating to another page', async () => {
  filterFrames
    .mockResolvedValueOnce({
      total_pages: 5,
      results: [{
        frame_id: 'frame-page-1',
        video_id: 'L24_topic.video-1',
        frame_idx: 1,
        timestamp_ms: 40,
      }],
    })
    .mockResolvedValueOnce({
      total_pages: 5,
      results: [{
        frame_id: 'frame-page-4',
        video_id: 'L24_topic.video-1',
        frame_idx: 4,
        timestamp_ms: 160,
      }],
    })
    .mockResolvedValueOnce({
      total_pages: 3,
      results: [{
        frame_id: 'frame-new-filter',
        video_id: 'L25_topic.video-1',
        frame_idx: 2,
        timestamp_ms: 80,
      }],
    });

  renderWorkspace({ isActive: false });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Page 4' })).toBeTruthy());

  fireEvent.click(screen.getByRole('button', { name: 'Page 4' }));
  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(2));
  expect(filterFrames.mock.calls[1][0]).toEqual(expect.objectContaining({
    pageId: 4,
    framesPerPage: expect.any(Number),
  }));

  fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'new filter' } });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));
  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(3));

  expect(filterFrames.mock.calls[2][0]).toEqual(expect.objectContaining({
    pageId: 1,
    filters: expect.objectContaining({ title: 'new filter' }),
  }));
});

test('changing page size reapplies the current session at page 1', async () => {
  filterFrames
    .mockResolvedValueOnce({ total_pages: 3, results: [] })
    .mockResolvedValueOnce({ total_pages: 2, results: [] });
  renderWorkspace({ isActive: false });
  fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'Boat' } });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));
  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(1));

  fireEvent.change(screen.getByLabelText('Frames per page'), {
    target: { value: '24' },
  });

  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(2));
  expect(filterFrames.mock.calls[1][0]).toEqual(expect.objectContaining({
    pageId: 1,
    framesPerPage: 24,
    filters: expect.objectContaining({ title: 'Boat' }),
  }));
});

test('freezes an Auto size for pagination and recalculates only a new filter', async () => {
  filterFrames
    .mockResolvedValueOnce({ total_pages: 2, results: [] })
    .mockResolvedValueOnce({ total_pages: 2, results: [] })
    .mockResolvedValueOnce({ total_pages: 2, results: [] });
  renderWorkspace({ isActive: false });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));
  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(1));
  const firstSize = filterFrames.mock.calls[0][0].framesPerPage;

  const viewport = document.querySelector('.filter-results');
  Object.defineProperty(viewport, 'clientWidth', { configurable: true, value: 1600 });
  Object.defineProperty(viewport, 'clientHeight', { configurable: true, value: 900 });
  fireEvent(window, new Event('resize'));
  expect(filterFrames).toHaveBeenCalledTimes(1);

  fireEvent.click(screen.getByRole('button', { name: 'Page 2' }));
  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(2));
  expect(filterFrames.mock.calls[1][0].framesPerPage).toBe(firstSize);

  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));
  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(3));
  expect(filterFrames.mock.calls[2][0]).toEqual(expect.objectContaining({
    pageId: 1,
    framesPerPage: expect.any(Number),
  }));
  expect(filterFrames.mock.calls[2][0].framesPerPage).toBeGreaterThan(firstSize);
});
