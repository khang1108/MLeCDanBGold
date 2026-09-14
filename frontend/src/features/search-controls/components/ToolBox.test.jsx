import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ToolBox from './ToolBox';
import AnswerWorkspaceProvider from '../../answer-workspace/contexts/AnswerWorkspaceContext';

const renderToolBox = (props) => render(
  <AnswerWorkspaceProvider connectedUserId="">
    <ToolBox {...props} />
  </AnswerWorkspaceProvider>,
);

describe('ToolBox component', () => {
  test('renders an inline Top-K number input without stepper controls, presets, or a slider', () => {
    const setTopK = jest.fn();
    renderToolBox({ topK: 20, setTopK });

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
    renderToolBox({ topK: 20, setTopK });

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
    renderToolBox({
      topK: 20,
      setTopK: jest.fn(),
      useDense: true,
      setUseDense,
      useBm25: true,
      setUseBm25,
    });

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
    renderToolBox({
      topK: 20,
      setTopK: jest.fn(),
      useDense: true,
      setUseDense: jest.fn(),
      useBm25: false,
      setUseBm25: jest.fn(),
    });

    expect(screen.getByRole('switch', { name: /use dense retrieval/i }).disabled).toBe(true);
    expect(screen.getByRole('switch', { name: /use bm25 retrieval/i }).disabled).toBe(false);
    expect(screen.queryByText(/at least one source/i)).toBeNull();
  });

  test('does not render the retired submission files panel in the Query sidebar', () => {
    renderToolBox({ topK: 20, setTopK: jest.fn() });

    expect(screen.queryByRole('region', { name: 'Shared submission files' })).toBeNull();
    expect(screen.queryByText('No Query Files')).toBeNull();
  });

  test('renders the shared answer workspace panel', () => {
    renderToolBox({ topK: 20, setTopK: jest.fn() });

    expect(screen.getByRole('region', { name: 'Answer workspace' })).toBeTruthy();
  });
});
