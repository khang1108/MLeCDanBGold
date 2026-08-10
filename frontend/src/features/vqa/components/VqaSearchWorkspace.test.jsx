import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFrames, searchVqa, suggestQueries } from '../../../api/search';
import VqaSearchWorkspace from './VqaSearchWorkspace';

jest.mock('../../../api/search');

beforeEach(() => {
  searchFrames.mockReset();
  searchVqa.mockReset();
  suggestQueries.mockReset();
});

const EVENT_PLACEHOLDER = 'Event query (/tkis, /vkis, /trake)...';
const QUESTION_PLACEHOLDER = 'Question (optional for VQA)...';

test('clicking Search sends both intents and renders grounded answer', async () => {
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
  fireEvent.change(screen.getByPlaceholderText(EVENT_PLACEHOLDER), {
    target: { value: 'A person reads a city sign' },
  });
  fireEvent.change(screen.getByPlaceholderText(QUESTION_PLACEHOLDER), {
    target: { value: 'Which city is shown?' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

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

test('a question always routes through VQA and strips prefixes', async () => {
  searchVqa.mockResolvedValueOnce({
    submissions: [], warnings: [], latency_ms: 12,
  });

  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  fireEvent.change(screen.getByPlaceholderText(EVENT_PLACEHOLDER), {
    target: { value: '/vkis a red vehicle passes' },
  });
  fireEvent.change(screen.getByPlaceholderText(QUESTION_PLACEHOLDER), {
    target: { value: 'What color is it?' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  await waitFor(() => expect(searchVqa).toHaveBeenCalledWith(
    expect.objectContaining({
      eventDescription: 'a red vehicle passes',
      question: 'What color is it?',
    }),
  ));
  expect(searchFrames).not.toHaveBeenCalled();
});

test.each([
  ['/kis a red vehicle passes', 'kis', 'a red vehicle passes'],
  ['/tkis blue car', 'kis', 'blue car'],
  ['/vkis green tree', 'vkis', 'green tree'],
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
      total: 4,
    },
  });

  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  fireEvent.change(screen.getByPlaceholderText(EVENT_PLACEHOLDER), {
    target: { value: description },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query, queryType, topK: 20 }),
  ));
  expect(searchVqa).not.toHaveBeenCalled();
});

test('requires a task prefix when the question is empty', () => {
  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  fireEvent.change(screen.getByPlaceholderText(EVENT_PLACEHOLDER), {
    target: { value: 'a red vehicle passes' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));

  expect(screen.getByRole('alert').textContent).toContain(
    'must start with /tkis, /vkis, or /trake',
  );
  expect(searchFrames).not.toHaveBeenCalled();
  expect(searchVqa).not.toHaveBeenCalled();
});

test('suggests queries preserving original prefix', async () => {
  suggestQueries.mockResolvedValueOnce({
    suggestions: [
      { suggestion_id: '1', query: 'a red vehicle driving fast', focus: 'action' }
    ]
  });

  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  fireEvent.change(screen.getByPlaceholderText(EVENT_PLACEHOLDER), {
    target: { value: '/vkis a red vehicle' },
  });
  
  fireEvent.click(screen.getByRole('button', { name: 'Suggest' }));
  
  await waitFor(() => expect(suggestQueries).toHaveBeenCalledWith(
    expect.objectContaining({ query: 'a red vehicle' }),
  ));
  
  const suggestionBtn = await screen.findByText('a red vehicle driving fast');
  fireEvent.click(suggestionBtn);
  
  expect(screen.getByPlaceholderText(EVENT_PLACEHOLDER).value).toBe('/vkis a red vehicle driving fast');
});
