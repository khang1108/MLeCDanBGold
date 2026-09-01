import React from 'react';
import { getPaginationItems } from '../filterPagination';

/** Compact pagination controls for the BE-owned Filter page count. */
const FilterPagination = ({ currentPage, totalPages, isLoading, onPageChange }) => {
  if (!Number.isInteger(totalPages) || totalPages <= 1) return null;

  const items = getPaginationItems(totalPages, currentPage);

  return (
    <nav className="filter-pagination" aria-label="Filter result pages">
      <button
        type="button"
        className="filter-page-btn filter-page-arrow"
        onClick={() => onPageChange(currentPage - 1)}
        disabled={isLoading || currentPage <= 1}
        aria-label="Previous page"
      >
        ‹
      </button>
      <div className="filter-page-list">
        {items.map((item) => {
          if (typeof item !== 'number') {
            return <span className="filter-page-ellipsis" key={item}>…</span>;
          }

          const isCurrent = item === currentPage;
          return (
            <button
              type="button"
              className={`filter-page-btn${isCurrent ? ' active' : ''}`}
              key={item}
              onClick={() => onPageChange(item)}
              disabled={isLoading || isCurrent}
              aria-current={isCurrent ? 'page' : undefined}
              aria-label={`Page ${item}`}
            >
              {item}
            </button>
          );
        })}
      </div>
      <button
        type="button"
        className="filter-page-btn filter-page-arrow"
        onClick={() => onPageChange(currentPage + 1)}
        disabled={isLoading || currentPage >= totalPages}
        aria-label="Next page"
      >
        ›
      </button>
    </nav>
  );
};

export default FilterPagination;
