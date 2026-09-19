import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import HypothesisAlternatives from './HypothesisAlternatives';

const mockAlternatives = [
  {
    alternative_id: 'alt_current',
    event_id: 'E1',
    representative_frame_id: 'f_curr',
    representative_frame_idx: 100,
    representative_timestamp_ms: 10000,
    is_current: true,
    score: 0.95,
    path: [],
  },
  {
    alternative_id: 'alt_later',
    event_id: 'E1',
    representative_frame_id: 'f_later',
    representative_frame_idx: 700,
    representative_timestamp_ms: 70000,
    is_current: false,
    score: 0.88,
    path: [],
  },
];

describe('HypothesisAlternatives', () => {
  test('previewing an alternative does not call mutation action', async () => {
    const onPreview = jest.fn();
    const onUse = jest.fn();

    render(
      <HypothesisAlternatives
        alternatives={mockAlternatives}
        onPreview={onPreview}
        onUse={onUse}
      />
    );

    // Click button with timestamp 70.00s
    const button = screen.getByRole('button', { name: /70\.00s/i });
    expect(button).toBeInTheDocument();
    fireEvent.click(button);

    expect(onPreview).toHaveBeenCalledWith(mockAlternatives[1]);
    expect(onUse).not.toHaveBeenCalled();
  });

  test('shows loading indicator when isLoading is true', () => {
    render(<HypothesisAlternatives isLoading={true} />);
    expect(screen.getByText(/loading complete-path alternatives/i)).toBeInTheDocument();
  });

  test('renders current badge and frame coordinates correctly', () => {
    render(
      <HypothesisAlternatives
        alternatives={mockAlternatives}
        activeAlternativeId="alt_later"
      />
    );

    expect(screen.getByText('Current')).toBeInTheDocument();
    expect(screen.getByText('Previewing')).toBeInTheDocument();
    expect(screen.getByText('#100')).toBeInTheDocument();
    expect(screen.getByText('#700')).toBeInTheDocument();
  });

  test('returns null when alternatives is empty', () => {
    const { container } = render(<HypothesisAlternatives alternatives={[]} />);
    expect(container.firstChild).toBeNull();
  });
});
