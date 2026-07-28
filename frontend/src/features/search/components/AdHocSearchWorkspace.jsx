import React, { useCallback, useState } from "react";
import { searchFrames } from "../../../api/search";
import FramesBox from "../../frames/components/FramesBox";
import ToolBox from "../../search-controls/components/ToolBox";

// Tab 2: Simple ad-hoc query workspace with top query bar and split options/results layout.
const AdHocSearchWorkspace = ({
  topK,
  setTopK,
  searchMode,
  setSearchMode,
  onFrameClick,
}) => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [latencyMs, setLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  const handleSubmit = useCallback(
    async (event) => {
      event?.preventDefault();
      const trimmed = query.trim();
      if (!trimmed || isSearching) return;

      setIsSearching(true);
      setError(null);
      try {
        const response = await searchFrames({
          query: trimmed,
          topK,
          searchMode,
        });
        setResults(response.results || []);
        setWarnings(response.warnings || []);
        setLatencyMs(response.latency_ms || null);
      } catch (err) {
        setError(err.message || "Failed to contact search API");
      } finally {
        setIsSearching(false);
      }
    },
    [isSearching, query, searchMode, topK],
  );

  const handleResetOptions = useCallback(() => {
    setTopK(20);
    setSearchMode("accurate");
  }, [setSearchMode, setTopK]);

  return (
    <div className="adhoc-workspace">
      <form className="adhoc-query-bar" onSubmit={handleSubmit}>
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
            className="input-text query-input-field"
            placeholder="Search frames by keyword or visual description..."
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            disabled={isSearching}
          />
        </div>
        <button
          type="submit"
          className="btn-primary query-submit-btn"
          disabled={isSearching || !query.trim()}
        >
          {isSearching ? "Searching..." : "Search"}
        </button>
      </form>

      <div className="adhoc-workspace-body">
        <aside className="adhoc-sidebar">
          <h3 className="adhoc-sidebar-title">Options</h3>
          <ToolBox
            topK={topK}
            setTopK={setTopK}
            searchMode={searchMode}
            setSearchMode={setSearchMode}
            onReset={handleResetOptions}
          />
        </aside>

        <section className="adhoc-results">
          <FramesBox
            results={results}
            isLoading={isSearching}
            error={error}
            latencyMs={latencyMs}
            warnings={warnings}
            onFrameClick={onFrameClick}
          />
        </section>
      </div>
    </div>
  );
};

export default AdHocSearchWorkspace;
