import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFrames, searchVqa } from '../../../api/search';
import VqaSearchWorkspace from './VqaSearchWorkspace';

jest.mock('../../../api/search');

beforeEach(() => {
  searchFrames.mockReset();
  searchVqa.mockReset();
});

test('clicking Search VQA sends both intents and renders grounded answer', async () => {
  searchVqa.mockResolvedValueOnce({
    submissions: [{
      rank: 1,
      video_id: 'video-2',
      frame_id: 'frame-2',
      frame_idx: 81,
      answer: 'Hồ Chí Minh',
      normalized_answer: 'hồ chí minh',
      joint_score: 0.86,
    }],
    warnings: [],
    latency_ms: 25,
  });

  render(<VqaSearchWorkspace topK={100} setTopK={jest.fn()} />);
  fireEvent.change(screen.getByLabelText('Event description'), {
    target: { value: 'A person reads a city sign' },
  });
  fireEvent.change(screen.getByLabelText('Question (optional)'), {
    target: { value: 'Which city is shown?' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search QA' }));

  await waitFor(() => expect(searchVqa).toHaveBeenCalledWith(
    expect.objectContaining({
      eventDescription: 'A person reads a city sign',
      question: 'Which city is shown?',
      topK: 100,
    }),
  ));
  expect(await screen.findByText('Hồ Chí Minh')).toBeTruthy();
  expect(screen.getByText(/video-2, frame 81/)).toBeTruthy();
});

test('a question always routes through VQA', async () => {
  searchVqa.mockResolvedValueOnce({
    submissions: [], warnings: [], latency_ms: 12,
  });

  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  fireEvent.change(screen.getByLabelText('Event description'), {
    target: { value: '/kis a red vehicle passes' },
  });
  fireEvent.change(screen.getByLabelText('Question (optional)'), {
    target: { value: 'What color is it?' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search QA' }));

  await waitFor(() => expect(searchVqa).toHaveBeenCalledWith(
    expect.objectContaining({
      eventDescription: '/kis a red vehicle passes',
      question: 'What color is it?',
    }),
  ));
  expect(searchFrames).not.toHaveBeenCalled();
});

test.each([
  ['/kis a red vehicle passes', 'kis', 'a red vehicle passes'],
  ['/trake person enters then leaves', 'trake', 'person enters then leaves'],
])('without a question routes %s through frame search', async (
  description,
  queryType,
  query,
) => {
  searchFrames.mockResolvedValueOnce({
    results: [],
    warnings: [],
    latency_ms: {
      query_processing: 0,
      query_encoding: 1,
      candidate_retrieval: 2,
      fusion: 0,
      reranking: 0,
      materialization: 1,
      total: 4,
    },
  });

  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  fireEvent.change(screen.getByLabelText('Event description'), {
    target: { value: description },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search KIS / TRAKE' }));

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query, queryType, topK: 20 }),
  ));
  expect(searchVqa).not.toHaveBeenCalled();
});

test('requires a task prefix when the question is empty', () => {
  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  fireEvent.change(screen.getByLabelText('Event description'), {
    target: { value: 'a red vehicle passes' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search KIS / TRAKE' }));

  expect(screen.getByRole('alert').textContent).toContain(
    'must start with /kis or /trake',
  );
  expect(searchFrames).not.toHaveBeenCalled();
  expect(searchVqa).not.toHaveBeenCalled();
});
