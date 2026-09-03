import React from 'react';

export const PaginationControls = ({
  page = 1,
  pageSize = 25,
  totalRows = 0,
  totalPages = 1,
  onChangePage,
  onChangePageSize,
  disabled = false,
}) => {
  const safeTotalPages = Math.max(1, totalPages);
  const canGoPrevious = page > 1 && !disabled;
  const canGoNext = page < safeTotalPages && !disabled;

  return (
    <div className="db-pagination-controls">
      <div className="db-pagination-nav">
        <button
          type="button"
          className="db-btn db-btn-pagination"
          disabled={!canGoPrevious}
          onClick={() => onChangePage(page - 1)}
        >
          Previous
        </button>
        <span className="db-pagination-info">
          Page {page} of {safeTotalPages} ({totalRows} total rows)
        </span>
        <button
          type="button"
          className="db-btn db-btn-pagination"
          disabled={!canGoNext}
          onClick={() => onChangePage(page + 1)}
        >
          Next
        </button>
      </div>

      <div className="db-pagination-size">
        <label htmlFor="db-page-size-select" className="db-size-label">
          Rows per page:
        </label>
        <select
          id="db-page-size-select"
          className="db-select"
          value={pageSize}
          disabled={disabled}
          onChange={(e) => onChangePageSize(Number(e.target.value))}
        >
          {[10, 25, 50, 100].map((size) => (
            <option key={size} value={size}>
              {size}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

export default PaginationControls;
