import React from "react";

const QUERY_TYPES = ["kis", "kisc", "vkis", "vqa", "trake"];

const QueryTypeBadge = ({ queryType, setQueryType }) => {
  const current = (queryType || "kis").toLowerCase();

  const handleCycle = () => {
    const currentIndex = QUERY_TYPES.indexOf(current);
    const nextIndex = (currentIndex + 1) % QUERY_TYPES.length;
    setQueryType(QUERY_TYPES[nextIndex]);
  };

  return (
    <button
      type="button"
      className={`query-type-badge type-${current}`}
      onClick={handleCycle}
      title={`Current Query Type: ${current.toUpperCase()}. Click to cycle (or press 1-5 in NORMAL mode).`}
    >
      <span className="query-type-value">{current.toUpperCase()}</span>
    </button>
  );
};

export default QueryTypeBadge;
