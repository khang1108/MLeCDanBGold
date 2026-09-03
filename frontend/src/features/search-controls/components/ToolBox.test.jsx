import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ToolBox from './ToolBox';

describe('ToolBox component', () => {
  test('renders an inline Top-K number input without stepper controls, presets, or a slider', () => {
    const setTopK = jest.fn();
    render(<ToolBox topK={20} setTopK={setTopK} />);

    expect(screen.getByText('Top-K results')).toBeTruthy();
    expect(screen.queryByRole('slider')).toBeNull();

    const numberInput = screen.getByLabelText(/top-k value/i);
    expect(numberInput).toBeTruthy();
    expect(numberInput.value).toBe('20');
    expect(numberInput.max).toBe('');
    expect(screen.queryByRole('button', { name: '10' })).toBeNull();
    expect(screen.queryByRole('button', { name: '20' })).toBeNull();
    expect(screen.queryByRole('button', { name: '50' })).toBeNull();
    expect(screen.queryByRole('button', { name: '100' })).toBeNull();
    expect(screen.queryByRole('button', { name: /increase top-k/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /decrease top-k/i })).toBeNull();
    expect(numberInput.closest('.toolbox-top-k-row')).toBeTruthy();
  });

  test('allows typing a custom value in direct input mode', () => {
    const setTopK = jest.fn();
    render(<ToolBox topK={20} setTopK={setTopK} />);

    const numberInput = screen.getByLabelText(/top-k value/i);
    fireEvent.change(numberInput, { target: { value: '10000' } });
    expect(setTopK).toHaveBeenCalledWith(10000);

    // Press Enter to commit
    fireEvent.keyDown(numberInput, { key: 'Enter', code: 'Enter' });
    expect(setTopK).toHaveBeenCalledWith(10000);
  });

  test('renders accessible Dense and BM25 switches', () => {
    const setUseDense = jest.fn();
    const setUseBm25 = jest.fn();
    render(
      <ToolBox
        topK={20}
        setTopK={jest.fn()}
        useDense
        setUseDense={setUseDense}
        useBm25
        setUseBm25={setUseBm25}
      />,
    );

    const dense = screen.getByRole('switch', { name: /use dense retrieval/i });
    const bm25 = screen.getByRole('switch', { name: /use bm25 retrieval/i });
    expect(dense.checked).toBe(true);
    expect(bm25.checked).toBe(true);

    fireEvent.click(dense);
    fireEvent.click(bm25);
    expect(setUseDense).toHaveBeenCalledWith(false);
    expect(setUseBm25).toHaveBeenCalledWith(false);
  });

  test('does not allow the only enabled retrieval source to be disabled', () => {
    render(
      <ToolBox
        topK={20}
        setTopK={jest.fn()}
        useDense
        setUseDense={jest.fn()}
        useBm25={false}
        setUseBm25={jest.fn()}
      />,
    );

    expect(screen.getByRole('switch', { name: /use dense retrieval/i }).disabled).toBe(true);
    expect(screen.getByRole('switch', { name: /use bm25 retrieval/i }).disabled).toBe(false);
    expect(screen.queryByText(/at least one source/i)).toBeNull();
  });

  test('keeps the submission files panel in the Query sidebar', () => {
    render(<ToolBox topK={20} setTopK={jest.fn()} />);

    expect(screen.getByRole('region', { name: 'Shared submission files' })).toBeTruthy();
    expect(screen.getByText('No Query Files')).toBeTruthy();
  });
});
