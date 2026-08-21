import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import QuerySuggestionsPanel from './QuerySuggestionsPanel';

describe('QuerySuggestionsPanel component', () => {
  test('renders empty prompt state when no suggestions are loaded', () => {
    render(<QuerySuggestionsPanel suggestions={[]} isLoading={false} />);
    expect(screen.getByText('5 Query Recommendations')).toBeTruthy();
    expect(screen.getByText(/Click/)).toBeTruthy();
  });

  test('renders 5 loading skeleton cards when isLoading is true', () => {
    const { container } = render(<QuerySuggestionsPanel suggestions={[]} isLoading={true} />);
    const skeletons = container.querySelectorAll('.suggest-card-skeleton');
    expect(skeletons).toHaveLength(5);
  });

  test('renders error state and handles retry', () => {
    const onRefresh = jest.fn();
    render(
      <QuerySuggestionsPanel
        suggestions={[]}
        isLoading={false}
        error="Network failure"
        onRefresh={onRefresh}
      />
    );
    expect(screen.getByText('Network failure')).toBeTruthy();
    const retryBtn = screen.getByRole('button', { name: /retry/i });
    fireEvent.click(retryBtn);
    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  test('renders up to 5 suggestions and calls onSelectSuggestion when clicked', () => {
    const onSelectSuggestion = jest.fn();
    const mockSuggestions = [
      'A person walking down the street',
      'A red car turning left',
      'People in a meeting room',
      'A chef cooking in kitchen',
      'A drone flying over skyscrapers',
    ];

    render(
      <QuerySuggestionsPanel
        suggestions={mockSuggestions}
        isLoading={false}
        error={null}
        onSelectSuggestion={onSelectSuggestion}
      />
    );

    expect(screen.getByText('Query Suggestions')).toBeTruthy();
    expect(screen.getAllByText('5').length).toBeGreaterThan(0);
    expect(screen.getByText('A person walking down the street')).toBeTruthy();
    expect(screen.getByText('A red car turning left')).toBeTruthy();
    expect(screen.getByText('People in a meeting room')).toBeTruthy();
    expect(screen.getByText('A chef cooking in kitchen')).toBeTruthy();
    expect(screen.getByText('A drone flying over skyscrapers')).toBeTruthy();

    fireEvent.click(screen.getByText('A red car turning left'));
    expect(onSelectSuggestion).toHaveBeenCalledWith('A red car turning left');
  });
});
