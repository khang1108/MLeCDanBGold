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


test('uses one keyword and unrestricted folder/video text scopes', async () => {
  filterFrames.mockResolvedValue({
    page_id: 1,
    total_pages: 0,
    total_results: 0,
    available_sources: ['caption'],
    results: [],
  });
  await renderWorkspace({ isActive: false });

  fireEvent.change(screen.getByLabelText('Keyword'), { target: { value: 'Áo đỏ' } });
  fireEvent.change(screen.getByLabelText('Folder'), { target: { value: 'custom_group' } });
  fireEvent.change(screen.getByLabelText('Video'), { target: { value: 'custom_video' } });
  fireEvent.click(screen.getByRole('button', { name: 'Search text' }));

  await waitFor(() => expect(filterFrames).toHaveBeenCalledTimes(1));
  expect(filterFrames).toHaveBeenCalledWith(expect.objectContaining({
    query: 'Áo đỏ',
    folderId: 'custom_group',
    videoId: 'custom_video',
    pageId: 1,
  }));
  expect(screen.queryByLabelText('Title')).toBeNull();
  expect(screen.queryByLabelText('Object 1, format name colon count')).toBeNull();
});


test('renders raw matched sources and opens the shared frame modal callback', async () => {
  const onFrameClick = jest.fn();
  const frame = {
    frame_id: 'L21_V001_keyframe_000001',
    video_id: 'L21_V001',
    folder_id: 'L21',
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
    total_pages: 1,
    total_results: 1,
    available_sources: ['caption', 'ocr', 'asr'],
    results: [frame],
  });
  await renderWorkspace({ isActive: false, onFrameClick });

  fireEvent.change(screen.getByLabelText('Keyword'), { target: { value: 'ao do' } });
  fireEvent.click(screen.getByRole('button', { name: 'Search text' }));

  expect(await screen.findByText(/ÁO ĐỎ/)).toBeTruthy();
  expect(screen.getByText(/Người mặc áo đỏ/)).toBeTruthy();
  fireEvent.click(screen.getByAltText(`Frame ${frame.frame_id}`));
  expect(onFrameClick).toHaveBeenCalledWith(frame);
});


test('keeps the shared submission files panel in the Filter sidebar', async () => {
  await renderWorkspace({ isActive: true });

  expect(screen.getByRole('region', { name: 'Shared submission files' })).toBeTruthy();
});
