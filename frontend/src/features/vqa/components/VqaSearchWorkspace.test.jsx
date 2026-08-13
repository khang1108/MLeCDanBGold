import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchFrames, searchTrake, searchVqa } from '../../../api/search';
import VqaSearchWorkspace, { progressiveSearchIdKey } from './VqaSearchWorkspace';

jest.mock('../../../api/search');

beforeEach(() => {
  searchFrames.mockReset();
  searchTrake.mockReset();
  searchVqa.mockReset();
  window.sessionStorage.clear();
});

const EVENT_PLACEHOLDER = 'Event query (/tkis, /vkis, /trake)...';
const QUESTION_PLACEHOLDER = 'Question (optional for VQA)...';

const submit = (eventDescription, question = '') => {
  fireEvent.change(screen.getByPlaceholderText(EVENT_PLACEHOLDER), {
    target: { value: eventDescription },
  });
  if (question) {
    fireEvent.change(screen.getByPlaceholderText(QUESTION_PLACEHOLDER), {
      target: { value: question },
    });
  }
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));
};

test('clicking Search sends both VQA intents and renders grounded answer', async () => {
  searchVqa.mockResolvedValueOnce({
    submissions: [{
      rank: 1,
      video_id: 'video-2',
      frame_id: 'frame-2',
      frame_idx: 81,
      answer: 'Hồ Chí Minh',
      normalized_answer: 'hồ chí minh',
      joint_score: 0.86,
      timestamp_ms: 3240,
      caption: 'A person reads a city sign.',
      thumbnail_url: 'http://example.test/frame-2.jpg',
    }],
    warnings: [],
    latency_ms: 25,
  });

  render(<VqaSearchWorkspace topK={100} setTopK={jest.fn()} />);
  submit('A person reads a city sign', 'Which city is shown?');

  await waitFor(() => expect(searchVqa).toHaveBeenCalledWith(
    expect.objectContaining({
      eventDescription: 'A person reads a city sign',
      question: 'Which city is shown?',
      topK: 100,
    }),
  ));
  expect(await screen.findByText('Hồ Chí Minh')).toBeTruthy();
  expect(screen.getByAltText('Frame frame-2').getAttribute('src')).toBe(
    'http://example.test/frame-2.jpg',
  );
});

test('a question always routes through VQA and strips prefixes', async () => {
  searchVqa.mockResolvedValueOnce({ submissions: [], warnings: [], latency_ms: 12 });
  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit('/vkis a red vehicle passes', 'What color is it?');

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
])('without a question routes %s through frame search', async (
  description,
  queryType,
  query,
) => {
  searchFrames.mockResolvedValueOnce({
    results: [], warnings: [], latency_ms: { total: 4 },
  });
  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit(description);

  await waitFor(() => expect(searchFrames).toHaveBeenCalledWith(
    expect.objectContaining({ query, queryType, topK: 20 }),
  ));
  expect(searchVqa).not.toHaveBeenCalled();
  expect(searchTrake).not.toHaveBeenCalled();
});

test('TRAKE uses the dedicated route and renders ordered path submissions', async () => {
  searchTrake.mockResolvedValueOnce({
    events: ['person enters', 'person leaves'],
    submissions: [{
      rank: 1,
      video_id: 'video-7',
      frame_ids: ['f10', 'f20'],
      frame_idxs: [10, 20],
    }],
    total_results: 1,
    warnings: [],
  });
  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit('/trake person enters -> person leaves');

  await waitFor(() => expect(searchTrake).toHaveBeenCalledWith(
    expect.objectContaining({ events: ['person enters', 'person leaves'], topK: 20 }),
  ));
  expect(searchFrames).not.toHaveBeenCalled();
  expect(await screen.findByText('Ordered TRAKE paths')).toBeTruthy();
  expect(screen.getByText('person enters: frame 10 (f10)')).toBeTruthy();
  expect(screen.getByText('person leaves: frame 20 (f20)')).toBeTruthy();
});

test('TRAKE requires at least two events without making a request', () => {
  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit('/trake only one event');
  expect(screen.getByRole('alert').textContent).toContain('at least two');
  expect(searchTrake).not.toHaveBeenCalled();
});

test('requires a task prefix when the question is empty', () => {
  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit('a red vehicle passes');
  expect(screen.getByRole('alert').textContent).toContain(
    'must start with /tkis, /vkis, or /trake',
  );
  expect(searchFrames).not.toHaveBeenCalled();
  expect(searchVqa).not.toHaveBeenCalled();
});

test('task-scoped search IDs never leak from KIS into VQA or TRAKE', async () => {
  const kisKey = progressiveSearchIdKey('kis');
  const vqaKey = progressiveSearchIdKey('vqa');
  window.sessionStorage.setItem(kisKey, 'kis-search-42');
  window.sessionStorage.setItem(vqaKey, 'vqa-search-9');
  searchVqa.mockResolvedValueOnce({
    search_id: 'vqa-search-9', submissions: [], warnings: [], latency_ms: 3,
  });
  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit('H1 cumulative clue', 'What is shown?');

  await waitFor(() => expect(searchVqa).toHaveBeenCalledWith(
    expect.objectContaining({ searchId: 'vqa-search-9' }),
  ));
  expect(searchVqa).not.toHaveBeenCalledWith(
    expect.objectContaining({ searchId: 'kis-search-42' }),
  );
  expect(searchTrake).not.toHaveBeenCalled();
});

test('New Question clears every task-scoped progressive ID', () => {
  const keys = ['kis', 'vkis', 'vqa'].map(progressiveSearchIdKey);
  keys.forEach((key, index) => window.sessionStorage.setItem(key, `search-${index}`));
  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  fireEvent.click(screen.getByRole('button', { name: 'New Question' }));
  keys.forEach((key) => expect(window.sessionStorage.getItem(key)).toBeNull());
});

test('410 clears only the active task ID and 409 explains how to reset', async () => {
  const vqaKey = progressiveSearchIdKey('vqa');
  const kisKey = progressiveSearchIdKey('kis');
  window.sessionStorage.setItem(vqaKey, 'expired-vqa');
  window.sessionStorage.setItem(kisKey, 'active-kis');
  const conflict = new Error('Progressive state conflict');
  conflict.status = 409;
  searchVqa.mockRejectedValueOnce(conflict);
  render(<VqaSearchWorkspace topK={20} setTopK={jest.fn()} />);
  submit('H1', 'What?');
  expect((await screen.findByRole('alert')).textContent).toContain('New Question');
  expect(window.sessionStorage.getItem(vqaKey)).toBe('expired-vqa');
  expect(window.sessionStorage.getItem(kisKey)).toBe('active-kis');

  const expired = new Error('Progressive state expired');
  expired.status = 410;
  searchVqa.mockRejectedValueOnce(expired);
  submit('H1', 'What?');
  await waitFor(() => expect(window.sessionStorage.getItem(vqaKey)).toBeNull());
  expect(window.sessionStorage.getItem(kisKey)).toBe('active-kis');
});
