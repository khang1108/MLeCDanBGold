import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import KisPanel from './KisPanel';

describe('KisPanel presentation component', () => {
  const defaultState = {
    draft: '',
    committedInputs: [],
    revision: 0,
    currentIntent: null,
    isSearching: false,
    error: null,
  };

  test('renders input with required placeholder "Search or add another clue…"', () => {
    render(<KisPanel sessionState={defaultState} />);
    expect(screen.getByPlaceholderText('Search or add another clue…')).toBeTruthy();
  });

  test('renders committed clue history when revision > 0', () => {
    const sessionState = {
      ...defaultState,
      committedInputs: ['A woman enters the kitchen', 'She takes a plate'],
      revision: 2,
    };
    render(<KisPanel sessionState={sessionState} />);
    expect(screen.getByText('Q1')).toBeTruthy();
    expect(screen.getByText('A woman enters the kitchen')).toBeTruthy();
    expect(screen.getByText('Q2')).toBeTruthy();
    expect(screen.getByText('She takes a plate')).toBeTruthy();
  });

  test('renders canonical query text, events, and entity chips from currentIntent', () => {
    const currentIntent = {
      revision: 2,
      inputs: ['A woman enters the kitchen', 'She takes a plate'],
      language: 'en',
      query_text: 'A woman enters the kitchen and takes a plate',
      entities: [
        { id: 'X1', kind: 'person', description: 'woman' },
        { id: 'X2', kind: 'object', description: 'plate' },
      ],
      events: [
        { id: 'E1', text: 'woman enters kitchen' },
        { id: 'E2', text: 'woman takes plate' },
      ],
      temporal_edges: [{ source: 'E1', relation: 'before', target: 'E2' }],
    };

    const sessionState = {
      ...defaultState,
      committedInputs: currentIntent.inputs,
      revision: 2,
      currentIntent,
    };

    render(<KisPanel sessionState={sessionState} />);

    // Canonical query text
    expect(screen.getByText('A woman enters the kitchen and takes a plate')).toBeTruthy();

    // Event list E1..En
    expect(screen.getByText(/E1:/)).toBeTruthy();
    expect(screen.getByText(/woman enters kitchen/)).toBeTruthy();
    expect(screen.getByText(/E2:/)).toBeTruthy();
    expect(screen.getByText(/woman takes plate/)).toBeTruthy();

    // Entity chips
    const entitiesContainer = screen.getByTestId('kis-intent-entities');
    expect(entitiesContainer).toBeTruthy();
    expect(entitiesContainer.textContent).toContain('X1:');
    expect(entitiesContainer.textContent).toContain('woman');
    expect(entitiesContainer.textContent).toContain('X2:');
    expect(entitiesContainer.textContent).toContain('plate');
  });

  test('calls onDraftChange when typing in the input', () => {
    const onDraftChange = jest.fn();
    render(<KisPanel sessionState={defaultState} onDraftChange={onDraftChange} />);
    const textarea = screen.getByPlaceholderText('Search or add another clue…');
    fireEvent.change(textarea, { target: { value: 'New clue' } });
    expect(onDraftChange).toHaveBeenCalledWith('New clue');
  });

  test('calls onSubmit on Enter (without Shift) and on button click', () => {
    const onSubmit = jest.fn();
    const sessionState = { ...defaultState, draft: 'New clue' };
    render(<KisPanel sessionState={sessionState} onSubmit={onSubmit} />);
    const textarea = screen.getByPlaceholderText('Search or add another clue…');

    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });
    expect(onSubmit).toHaveBeenCalledTimes(1);

    const submitBtn = screen.getByRole('button', { name: 'Search' });
    fireEvent.click(submitBtn);
    expect(onSubmit).toHaveBeenCalledTimes(2);
  });

  test('calls onReset when New Search button is clicked', () => {
    const onReset = jest.fn();
    render(<KisPanel sessionState={defaultState} onReset={onReset} />);
    const resetBtn = screen.getByRole('button', { name: /new search/i });
    fireEvent.click(resetBtn);
    expect(onReset).toHaveBeenCalledTimes(1);
  });
});
