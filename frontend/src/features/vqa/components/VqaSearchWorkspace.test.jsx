import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchVqa } from '../../../api/search';
import VqaSearchWorkspace from './VqaSearchWorkspace';

jest.mock('../../../api/search');

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
  fireEvent.change(screen.getByLabelText('Question'), {
    target: { value: 'Which city is shown?' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search VQA' }));

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
