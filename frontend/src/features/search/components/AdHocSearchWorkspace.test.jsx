import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import {
  getMiniChallengeCurrentTask,
  listMiniChallengeEvaluations,
  submitMiniChallengeFrame,
} from '../../../api/minichallenge';
import { searchFrames } from '../../../api/search';
import AdHocSearchWorkspace from './AdHocSearchWorkspace';

jest.mock('../../../api/minichallenge');
jest.mock('../../../api/search');

beforeEach(() => {
  jest.clearAllMocks();
});

test('clicking Search sends KIS request and renders retrieved frame', async () => {
  searchFrames.mockResolvedValueOnce({
    results: [{
      rank: 1,
      frame_id: 'frame-1',
      video_id: 'video-1',
      frame_idx: 42,
      timestamp_ms: 1200,
      caption: 'A red boat',
      thumbnail_url: null,
      frame_url: null,
      scores: { final: 0.9 },
    }],
    warnings: [],
    latency_ms: {
      total: 12,
      query_processing: 1,
      query_encoding: 2,
      candidate_retrieval: 5,
      fusion: 1,
      reranking: 0,
      materialization: 3,
    },
  });

  render(
    <AdHocSearchWorkspace
      topK={100}
      setTopK={jest.fn()}
      onFrameClick={jest.fn()}
    />,
  );
  fireEvent.change(screen.getByPlaceholderText(/Start with/), {
    target: { value: '/kis red boat' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query: 'red boat', topK: 100, queryType: 'kis' }),
  ));
  expect(await screen.findByText(/video-1 · frame 42/)).toBeTruthy();
  expect(screen.getAllByText('A red boat')).toHaveLength(2);
});

test('shows and wires the mini-challenge submit button after loading a task', async () => {
  listMiniChallengeEvaluations.mockResolvedValueOnce([{
    id: 'evaluation-1',
    name: 'Mini QA',
    status: 'ACTIVE',
    taskTemplates: [{
      name: 'tkis',
      taskGroup: 'TKIS Group',
      taskType: 'Textual KIS',
    }],
  }]);
  getMiniChallengeCurrentTask.mockResolvedValueOnce({
    name: 'tkis-00',
    taskGroup: 'TKIS Group',
    taskType: 'Textual KIS',
  });
  submitMiniChallengeFrame.mockResolvedValueOnce({
    status: true,
    submission: 'CORRECT',
    description: 'Accepted',
  });
  searchFrames.mockResolvedValueOnce({
    results: [{
      rank: 1,
      frame_id: 'frame-1',
      video_id: 'video-1',
      frame_idx: 42,
      timestamp_ms: 1200,
      caption: 'A red boat',
      thumbnail_url: null,
      frame_url: null,
      scores: { final: 0.9 },
    }],
    warnings: [],
    latency_ms: { total: 12 },
  });
  jest.spyOn(window, 'confirm').mockReturnValueOnce(true);

  render(
    <AdHocSearchWorkspace
      topK={100}
      setTopK={jest.fn()}
      onFrameClick={jest.fn()}
    />,
  );
  fireEvent.change(screen.getByPlaceholderText('Paste session token'), {
    target: { value: 'private-session' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Load evaluation' }));
  expect(await screen.findByText('tkis-00')).toBeTruthy();

  fireEvent.change(screen.getByPlaceholderText(/Start with/), {
    target: { value: '/kis red boat' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  const submit = await screen.findByRole('button', { name: 'Submit' });
  fireEvent.click(submit);
  await waitFor(() => expect(submitMiniChallengeFrame).toHaveBeenCalledWith({
    session: 'private-session',
    evaluationId: 'evaluation-1',
    frameId: 'frame-1',
    taskName: 'tkis',
    text: '',
  }));
});
