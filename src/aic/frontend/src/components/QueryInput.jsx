import React from 'react';

const QueryInput = ({ query, setQuery, onSearch }) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (onSearch) onSearch(query);
  };

  return (
    <form onSubmit={handleSubmit} className="query-form">
      <div className="query-input-wrapper">
        <svg
          className="query-search-icon"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.8}
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.603 10.602Z"
          />
        </svg>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question or enter a search query for the video frames... (e.g., 'red car speeding')"
          className="input-text query-input-field"
        />
      </div>
      <button type="submit" className="btn-primary query-submit-btn">
        Ask Question
      </button>
    </form>
  );
};

export default QueryInput;
