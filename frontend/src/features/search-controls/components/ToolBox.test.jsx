import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ToolBox from './ToolBox';

describe('ToolBox component', () => {
  test('renders Top-K number input and preset chips without any slider', () => {
    const setTopK = jest.fn();
    render(<ToolBox topK={20} setTopK={setTopK} onReset={jest.fn()} />);

    expect(screen.getByText('Top-K results')).toBeTruthy();
    expect(screen.queryByRole('slider')).toBeNull();

    const numberInput = screen.getByLabelText(/top-k value/i);
    expect(numberInput).toBeTruthy();
    expect(numberInput.value).toBe('20');

    expect(screen.getByRole('button', { name: '10' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '20' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '50' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '100' })).toBeTruthy();
  });

  test('allows typing a custom value in direct input mode', () => {
    const setTopK = jest.fn();
    render(<ToolBox topK={20} setTopK={setTopK} onReset={jest.fn()} />);

    const numberInput = screen.getByLabelText(/top-k value/i);
    fireEvent.change(numberInput, { target: { value: '35' } });
    expect(setTopK).toHaveBeenCalledWith(35);

    // Press Enter to commit
    fireEvent.keyDown(numberInput, { key: 'Enter', code: 'Enter' });
    expect(setTopK).toHaveBeenCalledWith(35);
  });

  test('stepper buttons increment and decrement value', () => {
    const setTopK = jest.fn();
    render(<ToolBox topK={20} setTopK={setTopK} onReset={jest.fn()} />);

    const increaseBtn = screen.getByLabelText(/increase top-k/i);
    fireEvent.click(increaseBtn);
    expect(setTopK).toHaveBeenCalledWith(21);

    const decreaseBtn = screen.getByLabelText(/decrease top-k/i);
    fireEvent.click(decreaseBtn);
    expect(setTopK).toHaveBeenCalledWith(20);
  });

  test('preset buttons set exact Top-K values', () => {
    const setTopK = jest.fn();
    render(<ToolBox topK={20} setTopK={setTopK} onReset={jest.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: '50' }));
    expect(setTopK).toHaveBeenCalledWith(50);

    fireEvent.click(screen.getByRole('button', { name: '100' }));
    expect(setTopK).toHaveBeenCalledWith(100);
  });

  test('clicking Reset Parameters triggers onReset callback', () => {
    const onReset = jest.fn();
    render(<ToolBox topK={50} setTopK={jest.fn()} onReset={onReset} />);

    fireEvent.click(screen.getByRole('button', { name: /reset parameters/i }));
    expect(onReset).toHaveBeenCalledTimes(1);
  });

  test('keeps the submission files panel in the Query sidebar', () => {
    render(<ToolBox topK={20} setTopK={jest.fn()} onReset={jest.fn()} />);

    expect(screen.getByRole('region', { name: 'Shared submission files' })).toBeTruthy();
    expect(screen.getByText('No Query Files')).toBeTruthy();
  });
});
