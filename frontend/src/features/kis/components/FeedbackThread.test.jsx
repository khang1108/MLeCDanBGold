import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import FeedbackThread from './FeedbackThread';

describe('FeedbackThread component', () => {
  const sampleMessages = [
    {
      id: 'm1',
      role: 'user',
      text: 'Find a woman in red dress',
      context: { eventId: 'E1', resultId: 'r1' },
    },
    {
      id: 'm2',
      role: 'assistant',
      text: 'Refined search query for E1.',
      scope: 'all_videos',
      changedEventIds: ['E1'],
    },
  ];

  test('renders user and assistant messages with context and scope chips', () => {
    render(
      <FeedbackThread
        messages={sampleMessages}
        canUndo={false}
        onUndo={jest.fn()}
      />,
    );

    expect(screen.getByText('Find a woman in red dress')).toBeTruthy();
    expect(screen.getByText('Refined search query for E1.')).toBeTruthy();
    expect(screen.getByText(/Context: E1/i)).toBeTruthy();
    expect(screen.getByText(/All videos/i)).toBeTruthy();
  });

  test('renders Undo button and fires onUndo callback when clicked', () => {
    const onUndo = jest.fn();
    render(
      <FeedbackThread
        messages={sampleMessages}
        canUndo={true}
        onUndo={onUndo}
      />,
    );

    const undoButton = screen.getByRole('button', { name: /undo/i });
    expect(undoButton).toBeTruthy();
    fireEvent.click(undoButton);
    expect(onUndo).toHaveBeenCalledTimes(1);
  });

  test('renders clarification notice properly', () => {
    const clarifyMessages = [
      {
        id: 'c1',
        role: 'assistant',
        text: 'Did you mean light blue or dark blue?',
        status: 'clarification',
      },
    ];

    render(
      <FeedbackThread
        messages={clarifyMessages}
        canUndo={false}
        onUndo={jest.fn()}
      />,
    );

    expect(screen.getByText('Did you mean light blue or dark blue?')).toBeTruthy();
    expect(screen.getByText(/clarification/i)).toBeTruthy();
  });
});
