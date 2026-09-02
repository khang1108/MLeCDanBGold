import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import FilterPageSize from './FilterPageSize';

test('offers accessible Auto, 12, 24, and 48 page sizes', () => {
  const onChange = jest.fn();
  render(<FilterPageSize value="auto" onChange={onChange} disabled={false} />);

  const selector = screen.getByLabelText('Frames per page');
  expect(Array.from(selector.options).map((option) => option.textContent)).toEqual([
    'Auto', '12', '24', '48',
  ]);
  fireEvent.change(selector, { target: { value: '24' } });
  expect(onChange).toHaveBeenCalledWith(24);
});

test('keeps page size disabled while a request is active', () => {
  render(<FilterPageSize value={12} onChange={jest.fn()} disabled />);

  expect(screen.getByLabelText('Frames per page').disabled).toBe(true);
});

