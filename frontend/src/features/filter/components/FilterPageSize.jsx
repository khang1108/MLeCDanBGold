import React from 'react';
import { FILTER_PAGE_SIZE_OPTIONS } from '../filterPagination';

/** Accessible selector for the bounded Filter page-size contract. */
const FilterPageSize = ({ value, onChange, disabled }) => (
  <label className="filter-page-size">
    <span>Frames per page</span>
    <select
      aria-label="Frames per page"
      value={String(value)}
      disabled={disabled}
      onChange={(event) => {
        const nextValue = event.target.value;
        onChange(nextValue === 'auto' ? 'auto' : Number(nextValue));
      }}
    >
      {FILTER_PAGE_SIZE_OPTIONS.map((option) => (
        <option key={option} value={String(option)}>
          {option === 'auto' ? 'Auto' : option}
        </option>
      ))}
    </select>
  </label>
);

export default FilterPageSize;

