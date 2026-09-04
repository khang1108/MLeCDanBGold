import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import FilterWorkspace from './FilterWorkspace';
import { SubmissionProvider } from '../../submission/contexts/SubmissionContext';
import { SubmissionDialogProvider } from '../../submission/contexts/SubmissionDialogContext';
import { filterFrames } from '../../../api/filter';
import { getSubmissionFiles } from '../../../api/workspace';


jest.mock('../../../api/filter');
jest.mock('../../../api/workspace', () => ({
  getSubmissionFiles: jest.fn().mockResolvedValue({ files: [] }),
  workspaceWebSocketUrl: jest.fn(() => 'ws://example.test/api/v1/workspace/ws'),
}));


const renderWorkspace = async (props = {}) => {
  const result = render(
    <SubmissionProvider>
      <SubmissionDialogProvider><FilterWorkspace {...props} /></SubmissionDialogProvider>
    </SubmissionProvider>,
  );
  await act(async () => Promise.resolve());
  return result;
};


beforeEach(() => {
  filterFrames.mockReset();
  getSubmissionFiles.mockResolvedValue({ files: [] });
});


test('restores separate evidence fields and sends Folder/Video only to the backend', async () => {
  filterFrames.mockResolvedValue({
    page_id: 1,
    frames_per_pages: 20,
    total_pages: 0,
    total_results: 0,
    results: [],
  });
  await renderWorkspace({ isActive: false });

  fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'Bản tin' } });
  fireEvent.change(screen.getByLabelText('ASR / Transcript'), { target: { value: 'xin chào' } });
  fireEvent.change(screen.getByLabelText('Caption'), { target: { value: 'áo đỏ' } });
  fireEvent.change(screen.getByLabelText('OCR'), { target: { value: 'biển báo' } });
  fireEvent.change(screen.getByLabelText('Object 1, format name colon count'), {
    target: { value: 'person: 2' },
  });
  fireEvent.change(screen.getByLabelText('Folder'), { target: { value: 'L21' } });
  fireEvent.change(screen.getByLabelText('Video'), { target: { value: 'L21_V001' } });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));

  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(1));
  expect(filterFrames).toHaveBeenCalledWith(expect.objectContaining({
    filters: expect.objectContaining({
      title: 'Bản tin',
      asr: 'xin chào',
      caption: 'áo đỏ',
      ocr: 'biển báo',
      objects: [expect.objectContaining({ value: 'person: 2' })],
    }),
    folderId: 'L21',
    videoId: 'L21_V001',
    pageId: 1,
  }));
  expect(screen.getByLabelText('Folder')).toBeTruthy();
  expect(screen.getByLabelText('Video')).toBeTruthy();
});


test('keeps repeatable object thresholds compact', async () => {
  await renderWorkspace({ isActive: false });

  fireEvent.change(screen.getByLabelText('Object 1, format name colon count'), {
    target: { value: 'chair: 4' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Add object filter' }));

  expect(screen.getByDisplayValue('chair: 4')).toBeTruthy();
  expect(screen.getByLabelText('Object 2, format name colon count')).toBeTruthy();
});


test('renders backend results without client-side scope filtering', async () => {
  const onFrameClick = jest.fn();
  const frame = {
    frame_id: 'L22_V002_keyframe_000001',
    video_id: 'L22_V002',
    folder_id: 'L22',
    frame_idx: 90,
    timestamp_ms: 3000,
    caption: 'A person wearing a red shirt',
    matches: {
      ocr: 'ÁO ĐỎ',
      asr: 'Người mặc áo đỏ đang đi vào.',
    },
  };
  filterFrames.mockResolvedValue({
    page_id: 1,
    frames_per_pages: 20,
    total_pages: 1,
    total_results: 1,
    results: [frame],
  });
  await renderWorkspace({ isActive: false, onFrameClick });

  fireEvent.change(screen.getByLabelText('Folder'), { target: { value: 'L21' } });
  fireEvent.change(screen.getByLabelText('Video'), { target: { value: 'L21_V001' } });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));

  expect(await screen.findByText(/ÁO ĐỎ/)).toBeTruthy();
  expect(filterFrames).toHaveBeenCalledWith(expect.objectContaining({
    folderId: 'L21',
    videoId: 'L21_V001',
  }));
  expect(screen.getByText(/Người mặc áo đỏ/)).toBeTruthy();
  fireEvent.click(screen.getByAltText(`Frame ${frame.frame_id}`));
  expect(onFrameClick).toHaveBeenCalledWith(frame);
});


test('reuses applied evidence predicates during pagination', async () => {
  filterFrames
    .mockResolvedValueOnce({
      page_id: 1,
      frames_per_pages: 20,
      total_pages: 2,
      total_results: 21,
      results: [],
    })
    .mockResolvedValueOnce({
      page_id: 2,
      frames_per_pages: 20,
      total_pages: 2,
      total_results: 21,
      results: [],
    });
  await renderWorkspace({ isActive: false });

  fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'boat' } });
  fireEvent.change(screen.getByLabelText('Folder'), { target: { value: 'L21' } });
  fireEvent.change(screen.getByLabelText('Video'), { target: { value: 'L21_V001' } });
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Page 2' })).toBeTruthy());
  fireEvent.click(screen.getByRole('button', { name: 'Page 2' }));

  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(2));
  expect(filterFrames.mock.calls[1][0]).toEqual(expect.objectContaining({
    filters: expect.objectContaining({ title: 'boat' }),
    folderId: 'L21',
    videoId: 'L21_V001',
    pageId: 2,
  }));
});


test('keeps the shared submission files panel in the Filter sidebar', async () => {
  await renderWorkspace({ isActive: true });

  expect(screen.getByRole('region', { name: 'Shared submission files' })).toBeTruthy();
});
