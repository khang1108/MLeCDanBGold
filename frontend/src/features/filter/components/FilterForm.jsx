import React, { useId } from 'react';


/** Collect one literal keyword shared by all text evidence sources. */
const FilterForm = ({ query, onChange, onSubmit, onReset, isLoading }) => {
  const inputId = useId();

  return (
    <form className="filter-form filter-keyword-form" onSubmit={onSubmit}>
      <label className="filter-field-label" htmlFor={inputId}>Keyword</label>
      <input
        id={inputId}
        aria-label="Keyword"
        className="input-text filter-input filter-keyword-input"
        value={query}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Search Title, Caption, OCR, ASR, or Objects"
        autoComplete="off"
      />
      <div className="filter-form-actions">
        <button type="button" className="btn-secondary filter-reset-btn" onClick={onReset}>
          Clear
        </button>
        <button
          type="submit"
          className="btn-primary filter-submit-btn"
          disabled={isLoading || !query.trim()}
        >
          {isLoading ? 'Searching…' : 'Search text'}
        </button>
      </div>
    </form>
  );
};


export default FilterForm;
